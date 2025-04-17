import time
import os
from typing import List, Dict, Tuple

# Core processing components
from .python_execution import try_python_execution, format_result
from .sql_execution import sql_fallback
from .e2e import e2e_response
from .voting import vote_on_solutions

# Utility components
from ..utils.prompt_utils import generate_prompt, create_reflection_prompt
from ..utils.code_utils import clean_code_response

# Data loading and client
from ..data.loader import load_table # Needed for logging
from ..clients.openai_client import OpenAIClient

# --- Helper Functions --- #

def select_model(attempt: int) -> str:
    """Selects a model based on the attempt number."""
    # Define a list of models to cycle through
    models = [
        "mistralai/codestral-2501",
        "meta-llama/llama-3.3-70b-instruct",
        "qwen/qwen-2.5-coder-32b-instruct",
        # Add more models here if desired
    ]
    return models[attempt % len(models)]

def log_attempt(log_file: str, question: str, dataset_name: str, is_sample: bool, attempt_num: int, model: str, code: str, result: str, success: bool):
    """Logs the details of a single solution attempt."""
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(f"-- Attempt {attempt_num} (Model: {model}) --\n")
            f.write(f"Question: {question}\n")
            f.write(f"Dataset: {dataset_name} (Sample: {is_sample})\n")
            try:
                columns = load_table(dataset_name, is_sample).columns.tolist()
                f.write(f"Columns: {columns}\n")
            except Exception as e:
                f.write(f"Columns: (Error loading: {e})\n")
            f.write(f"Generated Code:\n{code}\n")
            f.write(f"Execution Result: {str(result)}\n")
            f.write(f"Success: {success}\n")
            f.write("------\n")
    except Exception as e:
        print(f"Warning: Failed to write to attempt log file {log_file}: {e}")

# --- Core Python Solution Generation --- #

def get_python_solutions(question: str, dataset_name: str, is_sample: bool, ai_client: OpenAIClient, n_attempts: int = 1, log_dir: str = "logs") -> List[Dict]:
    """
    Generates and executes multiple Python code solutions using different models.
    Handles retries and reflection on failures.
    """
    print(f"\n--- Generating {n_attempts} Python Solution(s) for: {question[:50]}... ---")
    initial_solutions = []
    attempt_log_file = os.path.join(log_dir, "python_attempts_log.txt")

    # --- Initial Attempts --- #
    for i in range(n_attempts):
        attempt_start_time = time.time()
        model = select_model(i)
        print(f"Python Attempt {i + 1}/{n_attempts} using model {model}")

        code_response = ""
        result = None
        success = False

        try:
            # 1. Generate Prompt
            prompt_start_time = time.time()
            prompt = generate_prompt({"question": question, "dataset": dataset_name}, is_sample)
            # print(f"Time for generate_prompt: {time.time() - prompt_start_time:.4f}s")

            # 2. Generate Code Response
            response_start_time = time.time()
            raw_response = ai_client.generate_response(prompt, model=model)
            # print(f"Time for generate_response: {time.time() - response_start_time:.4f}s")

            # 3. Clean Code Response
            clean_start_time = time.time()
            code_response = clean_code_response(raw_response)
            # print(f"Time for clean_code_response: {time.time() - clean_start_time:.4f}s")
            # print(f"Cleaned code: {code_response}")

            # 4. Execute Code
            if code_response:
                execution_start_time = time.time()
                result, success = try_python_execution(code_response, dataset_name, is_sample)
                # print(f"Time for try_python_execution: {time.time() - execution_start_time:.4f}s")
            else:
                 result = "__CODE_GENERATION_FAILED__"
                 success = False
                 print("Skipping execution due to empty code response.")

        except Exception as e:
            print(f"Error during Python attempt {i + 1}: {e}")
            result = f"__ATTEMPT_ERROR__: {str(e)}"
            success = False
        finally:
            # Format result for consistency before storing/logging
            formatted_result = format_result(result)
            solution_data = {"code": code_response, "result": formatted_result, "success": success, "model": model}
            initial_solutions.append(solution_data)
            # Log this attempt
            log_attempt(attempt_log_file, question, dataset_name, is_sample, i + 1, model, code_response, formatted_result, success)
            print(f"Attempt {i+1} finished in {time.time() - attempt_start_time:.2f}s. Success: {success}")

    # --- Reflection and Retry on Failure (Optional) --- #
    successful_count = sum(s['success'] for s in initial_solutions)
    # Add reflection logic if desired, e.g., if successful_count < threshold
    # For now, we just return the initial attempts
    if successful_count < 1 and n_attempts > 0: # Example condition: if no success, try reflection
        print(f"\n--- No initial Python success. Attempting Reflection & Retry... ---")
        reflected_solutions = handle_failed_solutions(question, dataset_name, is_sample, ai_client, initial_solutions, n_attempts, log_dir)
        # Return combined or only reflected, depending on strategy
        # return initial_solutions + reflected_solutions
        return reflected_solutions # Example: Return only reflected if initial failed
    else:
        return initial_solutions

def handle_failed_solutions(question: str, dataset_name: str, is_sample: bool, ai_client: OpenAIClient, failed_solutions: List[Dict], n_attempts: int, log_dir: str = "logs") -> List[Dict]:
    """Generates new solutions based on reflecting on previous failures."""
    reflection_solutions = []
    reflection_log_file = os.path.join(log_dir, "reflection_attempts_log.txt")

    try:
        df = load_table(dataset_name, is_sample)
        # Filter tracebacks correctly, ensuring 'result' is a string error message
        tracebacks = [str(s['result']) for s in failed_solutions if not s['success'] and isinstance(s.get('result'), str)]
        if not tracebacks:
             print("No specific error messages found in failed solutions for reflection.")
             return [] # Cannot reflect without error info

        reflection_prompt = create_reflection_prompt(question, df, tracebacks)
    except Exception as setup_e:
        print(f"Error setting up reflection: {setup_e}")
        return []

    print(f"Generating {n_attempts} solutions via reflection...")
    for j in range(n_attempts):
        attempt_start_time = time.time()
        model = select_model(j) # Cycle through models for reflection attempts too
        print(f"Reflection Attempt {j + 1}/{n_attempts} using model {model}")

        code_response = ""
        result = None
        success = False

        try:
            raw_response = ai_client.generate_response(reflection_prompt, model=model)
            code_response = clean_code_response(raw_response) # Use the same cleaning
            # print(f"Reflected code: {code_response}")

            if code_response:
                result, success = try_python_execution(code_response, dataset_name, is_sample)
            else:
                result = "__CODE_GENERATION_FAILED__"
                success = False
                print("Skipping execution due to empty reflected code response.")

        except Exception as e:
            print(f"Error during Reflection attempt {j + 1}: {e}")
            result = f"__ATTEMPT_ERROR__: {str(e)}"
            success = False
        finally:
            formatted_result = format_result(result)
            solution_data = {"code": code_response, "result": formatted_result, "success": success, "model": f"{model}-Reflected"}
            reflection_solutions.append(solution_data)
            log_attempt(reflection_log_file, question, dataset_name, is_sample, j + 1, model, code_response, formatted_result, success)
            print(f"Reflection Attempt {j+1} finished in {time.time() - attempt_start_time:.2f}s. Success: {success}")

    return reflection_solutions

# --- Main Processing Function --- #

def process_question(dataset_name: str, question: str, is_sample: bool, ai_client: OpenAIClient, config: Dict) -> Tuple[str, List[Dict]]:
    """
    Main processing function that orchestrates the solution generation and selection.
    Configurable parameters passed via `config` dictionary.
    """
    pipeline_start_time = time.time()
    print(f"\n{'='*20} Processing Question {'='*20}")
    print(f"Dataset: {dataset_name}, Sample: {is_sample}")
    print(f"Question: {question}")

    # --- Configuration --- #
    n_python_attempts = config.get('n_python_attempts', 1)
    enable_reflection = config.get('enable_reflection', True)
    enable_sql_fallback = config.get('enable_sql_fallback', True)
    n_sql_attempts = config.get('n_sql_attempts', 1)
    enable_e2e = config.get('enable_e2e', True)
    log_dir = config.get('log_dir', 'logs')

    all_solutions = []

    # --- 1. Python Code Generation --- #
    python_solutions = get_python_solutions(question, dataset_name, is_sample, ai_client, n_python_attempts, log_dir)
    all_solutions.extend(python_solutions)

    # --- 2. Reflection (if enabled and needed) --- #
    successful_python = [s for s in python_solutions if s['success']]
    if enable_reflection and not successful_python and python_solutions:
        print("\n--- Triggering Reflection due to no initial Python success ---")
        reflection_solutions = handle_failed_solutions(question, dataset_name, is_sample, ai_client, python_solutions, n_python_attempts, log_dir)
        all_solutions.extend(reflection_solutions)

    # --- 3. SQL Fallback (if enabled) --- #
    sql_solutions = []
    if enable_sql_fallback:
        # Pass only initially failed *Python* solutions to SQL prompt generator
        failed_python_solutions = [s for s in python_solutions if not s['success']]
        sql_solutions = sql_fallback(dataset_name, question, is_sample, failed_python_solutions, ai_client, n_sql_attempts, os.path.join(log_dir, "sql_fallback_log.txt"))
        all_solutions.extend(sql_solutions)

    # --- 4. E2E Response (if enabled) --- #
    e2e_solutions = []
    if enable_e2e:
        e2e_solutions = e2e_response(question, dataset_name, is_sample, ai_client)
        all_solutions.extend(e2e_solutions)

    # --- 5. Voting --- #
    print(f"\n--- Final Voting Stage --- ({len(all_solutions)} total solutions gathered)")
    successful_solutions = [s for s in all_solutions if s.get("success")]
    print(f"Voting on {len(successful_solutions)} successful solutions.")

    final_result = "__NO_SOLUTION_FOUND__"
    if successful_solutions:
        voting_start_time = time.time()
        final_result = vote_on_solutions(successful_solutions, question, dataset_name, is_sample, ai_client, log_dir)
        print(f"Time for vote_on_solutions: {time.time() - voting_start_time:.2f} seconds")
        # Ensure final result is a simple string
        if isinstance(final_result, list):
            final_result = str(final_result[0] if len(final_result) == 1 else final_result)
        elif not isinstance(final_result, str):
             final_result = str(final_result)
        # Basic cleaning for the final voted answer
        final_result = final_result.strip().replace('"', '').replace("'", '')
    else:
        print("No successful solutions found across all methods.")
        # Optionally, check for any result even if marked as failed
        if all_solutions:
             final_result = f"__NO_SUCCESSFUL_SOLUTION__ ({all_solutions[0]['result']})" # Example: Show first failed result

    print(f"Final Result: {final_result}")
    print(f"Total processing time for question: {time.time() - pipeline_start_time:.2f} seconds")
    print(f"{'='*58}")

    # Return the final chosen result and the log of all solutions attempted
    return final_result, all_solutions 
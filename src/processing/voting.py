import re
from typing import List, Dict
import os

# Placeholder imports
from ..utils.prompt_utils import generate_voting_prompt
from ..clients.openai_client import OpenAIClient # For type hint

def vote_on_solutions(solutions: List[Dict], question: str, dataset_name: str, is_sample: bool, ai_client: OpenAIClient, log_dir: str = "logs") -> str:
    """
    Uses an LLM to vote on the best solution among successful attempts.
    """
    successful_solutions = [s for s in solutions if s.get("success")]

    if not successful_solutions:
        print("Warning: No successful solutions to vote on.")
        # Fallback: Return the result of the first solution attempt, if any
        return solutions[0]["result"] if solutions else "__NO_VALID_SOLUTION__"

    if len(successful_solutions) == 1:
        print("Only one successful solution, using its result.")
        return successful_solutions[0]["result"]

    print(f"\n--- Voting on {len(successful_solutions)} successful solutions for: {question[:50]}... ---")

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "voting_log.txt")

    # Generate voting prompt
    voting_prompt = generate_voting_prompt(successful_solutions, question, dataset_name, is_sample)

    try:
        # Get voting response from LLM
        # Consider making the voting model configurable
        response = ai_client.generate_response(voting_prompt, model="meta-llama/llama-3.3-70b-instruct").strip()
        print("Voting response received from LLM.")

        # Log the detailed prompt and response for debugging
        try:
            with open(log_file_path, "a", encoding='utf-8') as f:
                trimmed_voting_prompt = f'Given the following solutions and their results for the task: "{question}"\n\n' + \
                    '\n'.join([f'Solution Number {i+1}:\nCode: {r["code"]} Answer: {str(r["result"])[:100]} (may be truncated)\n'
                              for i, r in enumerate(successful_solutions)])
                f.write(f"-------------------------\nVoting Prompt:\n{trimmed_voting_prompt}\n\nRaw LLM Response:\n{response}\n")
        except Exception as log_e:
            print(f"Warning: Failed to write to voting log file {log_file_path}: {log_e}")

        # --- Extract the chosen solution number --- 
        chosen_solution_index = -1 # Default to invalid index

        # 1. Look for "ANSWER: <number>"
        answer_match = re.search(r'ANSWER:\s*(\d+)', response, re.IGNORECASE)
        if answer_match:
            try:
                 chosen_solution_index = int(answer_match.group(1)) - 1 # Convert to 0-based index
                 print(f"Voting result parsed (ANSWER: method): Solution #{chosen_solution_index + 1}")
            except ValueError:
                print("Warning: Found 'ANSWER:' but number parsing failed.")
        
        # 2. If not found, look for the last number in the response as a fallback
        if chosen_solution_index == -1:
            numeric_matches = re.findall(r'\b(\d+)\b', response)
            if numeric_matches:
                try:
                    chosen_solution_index = int(numeric_matches[-1]) - 1 # Use last number found
                    print(f"Voting result parsed (last number fallback): Solution #{chosen_solution_index + 1}")
                except ValueError:
                     print("Warning: Found numbers but last one failed parsing.")

        # --- Validate and return the chosen solution's result --- 
        if 0 <= chosen_solution_index < len(successful_solutions):
            final_result = successful_solutions[chosen_solution_index]["result"]
            print(f"Selected solution {chosen_solution_index + 1} with result: {str(final_result)[:100]}")
            # Log the chosen index
            try:
                 with open(log_file_path, "a", encoding='utf-8') as f:
                     f.write(f"Chosen Solution Index: {chosen_solution_index}\nResult: {final_result}\n-------------------------\n")
            except Exception as log_e:
                 print(f"Warning: Failed to write chosen index to voting log: {log_e}")
            return final_result
        else:
            print("Warning: Could not reliably determine chosen solution number from response. Defaulting to the first successful solution.")
            # Log the failure to parse
            try:
                 with open(log_file_path, "a", encoding='utf-8') as f:
                     f.write(f"Chosen Solution Index: FAILED_TO_PARSE (Defaulting to 0)\nResult: {successful_solutions[0]['result']}\n-------------------------\n")
            except Exception as log_e:
                 print(f"Warning: Failed to write parse failure to voting log: {log_e}")
            return successful_solutions[0]["result"] # Fallback to first successful result

    except Exception as e:
        print(f"Error during voting process: {e}")
        print("Defaulting to the first successful solution due to voting error.")
        # Log the error
        try:
             with open(log_file_path, "a", encoding='utf-8') as f:
                 f.write(f"VOTING_ERROR: {e}\nDefaulting to Solution 0\nResult: {successful_solutions[0]['result']}\n-------------------------\n")
        except Exception as log_e:
             print(f"Warning: Failed to write voting error to log: {log_e}")
        return successful_solutions[0]["result"] # Fallback on error 
import os
import time
import argparse
from tqdm import tqdm
import json

# Import core components from the src package
from src.data.loader import load_data
from src.clients.openai_client import OpenAIClient
from src.processing.pipeline import process_question
from src.evaluation.evaluate import evaluate_responses

def main(config):
    """Main function to run the question answering pipeline."""
    run_start_time = time.time()

    # --- Configuration --- #
    data_dir = config.get('data_dir', './data')
    log_dir = config.get('log_dir', 'logs')
    results_file = config.get('results_file', 'solution_results.txt')
    results_log_file = config.get('results_log_file', 'solution_results_logs.txt')
    dataset_name = config.get('dataset_name', 'semeval')
    dataset_split = config.get('dataset_split', 'dev')
    is_sample = config.get('is_sample', False) # Use full dataset by default
    start_index = config.get('start_index', None) # Process from beginning if not specified
    max_questions = config.get('max_questions', None) # Process all if not specified
    batch_size = config.get('batch_size', 1) # Process one question at a time by default

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    print("--- Configuration ---")
    print(f"Data Directory: {data_dir}")
    print(f"Log Directory: {log_dir}")
    print(f"Results File: {results_file}")
    print(f"Dataset: {dataset_name}/{dataset_split}")
    print(f"Use Sample Data: {is_sample}")
    print(f"Start Index: {start_index}")
    print(f"Max Questions: {max_questions}")
    print(f"Batch Size: {batch_size}")
    print("Pipeline Config:")
    for key, value in config['pipeline_config'].items():
        print(f"  {key}: {value}")
    print("---------------------")

    # --- Initialization --- #
    try:
        ai_client = OpenAIClient() # Initializes based on .env file
        dataset = load_data(name=dataset_name, split=dataset_split, data_dir=data_dir)
    except Exception as e:
        print(f"Fatal Error during initialization: {e}")
        return

    # --- Determine processing range --- #
    processed_count = 0
    if start_index is None:
        # If no start index, check results file to resume
        if os.path.exists(results_file):
            try:
                with open(results_file, "r", encoding="utf-8") as f:
                    processed_count = sum(1 for _ in f)
                print(f"Resuming from line {processed_count + 1} based on {results_file}")
                start_index = processed_count
            except Exception as e:
                print(f"Warning: Could not read {results_file} to determine resume point: {e}. Starting from index 0.")
                start_index = 0
        else:
            start_index = 0
            print(f"Results file {results_file} not found. Starting from index 0.")
    else:
        print(f"Starting from specified index {start_index}")

    total_questions = len(dataset)
    actual_start_index = min(start_index, total_questions)

    if max_questions is None:
        # Process all questions from start index
        actual_end_index = total_questions
    else:
        # Process up to max_questions or end of dataset, whichever comes first
        actual_end_index = min(actual_start_index + max_questions, total_questions)

    if actual_start_index >= actual_end_index:
        print("No questions to process in the specified range.")
        return

    print(f"Processing questions from index {actual_start_index} to {actual_end_index - 1} (Total: {actual_end_index - actual_start_index}).")
    dataset_slice = dataset.select(range(actual_start_index, actual_end_index))

    # --- Batch Processing --- #
    responses = []
    all_detailed_logs = [] # Store detailed logs for each question

    # Open results file in append mode
    try:
        with open(results_file, "a", encoding="utf-8") as res_f:
            for i in range(0, len(dataset_slice), batch_size):
                batch = dataset_slice.select(range(i, min(i + batch_size, len(dataset_slice))))
                print(f"\nProcessing Batch {i // batch_size + 1} (Indices {actual_start_index + i} to {actual_start_index + min(i + batch_size, len(dataset_slice)) - 1})...")

                for row_index, row in enumerate(batch):
                    question_global_index = actual_start_index + i + row_index
                    print(f"---\nProcessing Question Index: {question_global_index}")
                    try:
                        response, detailed_log = process_question(
                            dataset_name=row["dataset"],
                            question=row["question"],
                            is_sample=is_sample,
                            ai_client=ai_client,
                            config=config['pipeline_config']
                        )
                        # Write response immediately to results file
                        res_f.write(f"{str(response).replace(chr(10), ' ')}\n")
                        res_f.flush() # Ensure it's written to disk
                        responses.append(response)
                        all_detailed_logs.append({"index": question_global_index, "log": detailed_log})

                    except Exception as e:
                        print(f"FATAL ERROR processing question index {question_global_index}: {e}")
                        # Option: write an error marker to the results file
                        res_f.write("__PROCESSING_ERROR__\n")
                        res_f.flush()
                        responses.append("__PROCESSING_ERROR__")
                        all_detailed_logs.append({"index": question_global_index, "log": [{"error": str(e)}]})
                        # Decide whether to break or continue on error
                        if config.get('stop_on_error', True):
                            print("Stopping execution due to error.")
                            break # Stop processing this batch
                        else:
                            print("Continuing to next question despite error.")
                else:
                     continue # Only executed if the inner loop did NOT break
                break # Only executed if the inner loop DID break

    except IOError as e:
        print(f"Fatal Error: Could not open or write to results file {results_file}: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during batch processing: {e}")
        # Consider logging this error
        return

    print(f"\nFinished processing {len(responses)} questions.")

    # --- Evaluation --- #
    if responses:
        print("\n--- Evaluating Results ---")
        try:
            accuracy, eval_logs = evaluate_responses(responses, dataset_slice, is_sample=is_sample)
            print(f"Overall Accuracy on processed questions: {accuracy:.4f}")

            # Save evaluation logs
            with open(results_log_file, "a", encoding="utf-8") as log_f:
                 log_f.write(f"\n--- Evaluation Run: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                 log_f.write(f"Dataset: {dataset_name}/{dataset_split}, Sample: {is_sample}\n")
                 log_f.write(f"Processed Indices: {actual_start_index} - {actual_end_index - 1}\n")
                 log_f.write(f"Accuracy: {accuracy:.4f}\n")
                 log_f.write(f"Configuration: {json.dumps(config)}\n")
                 log_f.write("Incorrect Answers Log:\n")
                 for resp, truth, semantic, quest, verdict in eval_logs:
                     if not verdict:
                        log_f.write(f"  Index: {actual_start_index + eval_logs.index((resp, truth, semantic, quest, verdict))} | Response: {resp} | Truth: {truth} | Type: {semantic} | Question: {quest}\n")
        except Exception as e:
            print(f"Error during evaluation: {e}")
    else:
        print("No responses generated, skipping evaluation.")

    # --- Save Detailed Logs (Optional) ---
    detailed_log_file = os.path.join(log_dir, "detailed_solutions_log.json")
    try:
        with open(detailed_log_file, "a", encoding="utf-8") as f:
            for entry in all_detailed_logs:
                 json.dump(entry, f, ensure_ascii=False)
                 f.write("\n")
        print(f"Detailed solution logs saved to {detailed_log_file}")
    except Exception as e:
        print(f"Warning: Failed to save detailed logs: {e}")

    print(f"\nTotal script execution time: {time.time() - run_start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tabular QA Pipeline.")

    # --- Command Line Arguments --- #
    parser.add_argument("--config", type=str, default="config.json", help="Path to JSON configuration file.")
    # Add command-line overrides for specific config options if needed
    parser.add_argument("--start_index", type=int, help="Force start processing from this index (overrides resume logic). Example: 0")
    parser.add_argument("--max_questions", type=int, help="Maximum number of questions to process. Example: 10")
    parser.add_argument("--is_sample", action='store_true', help="Use sample dataset tables instead of full tables.")
    parser.add_argument("--batch_size", type=int, help="Number of questions to process per batch. Example: 5")

    args = parser.parse_args()

    # --- Load Configuration --- #
    config = {
        # Default values
        "data_dir": "./data",
        "log_dir": "./logs",
        "results_file": "solution_results.txt",
        "results_log_file": "solution_results_logs.txt",
        "dataset_name": "semeval",
        "dataset_split": "dev",
        "is_sample": False,
        "start_index": None,
        "max_questions": None,
        "batch_size": 1,
        "stop_on_error": True,
        "pipeline_config": {
            "n_python_attempts": 1,
            "enable_reflection": True,
            "enable_sql_fallback": True,
            "n_sql_attempts": 1,
            "enable_e2e": True
        }
    }

    if os.path.exists(args.config):
        print(f"Loading configuration from {args.config}...")
        try:
            with open(args.config, 'r') as f:
                loaded_config = json.load(f)
                # Merge loaded config with defaults, prioritizing loaded values
                def merge_configs(default, loaded):
                    for key, value in loaded.items():
                        if isinstance(value, dict) and key in default and isinstance(default[key], dict):
                            merge_configs(default[key], value)
                        else:
                            default[key] = value
                    return default
                config = merge_configs(config, loaded_config)
        except Exception as e:
            print(f"Warning: Failed to load or parse config file {args.config}: {e}. Using default config.")
    else:
        print(f"Config file {args.config} not found. Using default config.")

    # --- Override config with command-line arguments --- #
    if args.start_index is not None:
        config['start_index'] = args.start_index
    if args.max_questions is not None:
        config['max_questions'] = args.max_questions
    if args.is_sample:
        config['is_sample'] = True
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size

    # --- Run Main Function --- #
    main(config) 
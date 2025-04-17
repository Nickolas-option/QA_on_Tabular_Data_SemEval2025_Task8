import time
import pandas as pd
from typing import List, Tuple, Any
from datasets import Dataset

def evaluate_responses(responses: List[str], dataset: Dataset, is_sample: bool = True) -> Tuple[float, List]:
    """Evaluates the generated responses against the ground truth in the dataset."""
    start_time = time.time()
    correct = 0
    answer_key = "sample_answer" if is_sample else "answer"

    if answer_key not in dataset.column_names:
        print(f"Error: Answer key '{answer_key}' not found in dataset columns: {dataset.column_names}")
        return (0.0, [])

    truths = dataset[answer_key]
    logs = []

    if not responses:
        print("Warning: No responses provided for evaluation.")
        return (0.0, logs)

    if len(responses) != len(truths):
        print(f"Warning: Number of responses ({len(responses)}) does not match number of truths ({len(truths)}). Evaluating partial results.")
        # Adjust evaluation range if lengths mismatch
        eval_count = min(len(responses), len(truths))
        responses = responses[:eval_count]
        truths = truths[:eval_count]
        # Adjust other dataset columns used in zip if necessary
        eval_indices = range(eval_count)
        dataset_subset = dataset.select(eval_indices)
    else:
        dataset_subset = dataset # Use the full dataset if lengths match

    # Ensure required columns exist for logging
    required_log_cols = ["type", "question"]
    if not all(col in dataset_subset.column_names for col in required_log_cols):
         print(f"Warning: Missing required columns for logging ({required_log_cols}). Log details might be incomplete.")
         # Adjust logging tuple creation if columns are missing

    for i, (response, truth) in enumerate(zip(responses, truths)):
        # Safely get optional log info
        semantic = dataset_subset[i].get("type", "Unknown")
        question = dataset_subset[i].get("question", "Unknown")

        is_correct = compare_results(response, truth, semantic)

        if is_correct:
            correct += 1
        else:
            # Print incorrect responses for debugging
            print(f"Incorrect: Response=\"{response}\" || Truth=\"{truth}\" || Type='{semantic}' || Question=\"{question}\" || Verdict={is_correct}")

        logs.append((response, truth, semantic, question, is_correct))

    total_evaluated = len(truths)
    if total_evaluated == 0:
        print("No responses were evaluated.")
        accuracy = 0.0
    else:
        accuracy = correct / total_evaluated
        print(f"Evaluation finished in {time.time() - start_time:.2f} seconds. Accuracy: {accuracy:.4f} ({correct}/{total_evaluated})")

    if not logs:
        print("No logs were generated during evaluation.")

    return (accuracy, logs)

def compare_results(value: Any, truth: Any, semantic: str) -> bool:
    """Compares a single predicted value against the truth based on semantic type."""
    STRIP_CHARS = "[]'\" `"
    semantic = str(semantic).strip().lower()
    result = False

    # Standardize None/empty representations
    value_str = str(value).strip(STRIP_CHARS)
    truth_str = str(truth).strip(STRIP_CHARS)

    if value is None and truth is None:
        return True
    if value is None or truth is None:
        # Handle cases like comparing None to "None" or empty string
        if (value is None and not truth_str) or (truth is None and not value_str):
             return True
        return False

    try:
        if semantic == "boolean":
            # Flexible boolean comparison (True/False, true/false, 1/0)
            value_bool = value_str.lower() in ['true', '1']
            truth_bool = truth_str.lower() in ['true', '1']
            result = value_bool == truth_bool
        elif semantic == "category":
            # Direct string comparison after stripping
            result = value_str.lower() == truth_str.lower()
            # Attempt date comparison as a fallback if direct match fails
            if not result:
                try:
                    # Use errors='coerce' to handle unparseable dates gracefully
                    value_date = pd.to_datetime(value_str, errors='coerce').date()
                    truth_date = pd.to_datetime(truth_str, errors='coerce').date()
                    if value_date and truth_date:
                        result = value_date == truth_date
                    elif not value_str and not truth_str: # Both empty strings after strip
                        result = True
                except (ValueError, TypeError):
                    # If date conversion fails, rely on initial string comparison result
                    pass
        elif semantic == "number":
            try:
                # Clean string to keep only digits, decimal point, and minus sign
                value_cleaned = ''.join(char for char in value_str if char.isdigit() or char in ['.', '-'])
                truth_cleaned = ''.join(char for char in truth_str if char.isdigit() or char in ['.', '-'])
                # Handle potential empty strings after cleaning
                if not value_cleaned or not truth_cleaned:
                    result = value_cleaned == truth_cleaned # True if both are empty
                else:
                    result = round(float(value_cleaned), 2) == round(float(truth_cleaned), 2)
            except (ValueError, TypeError):
                result = False # Failed to convert to float
        elif semantic == "list[category]":
            try:
                # Split, strip, and filter empty strings
                value_list = [item.strip(STRIP_CHARS).lower() for item in value_str.split(',') if item.strip(STRIP_CHARS)]
                truth_list = [item.strip(STRIP_CHARS).lower() for item in truth_str.split(',') if item.strip(STRIP_CHARS)]
                # Compare as sets for order independence
                result = set(value_list) == set(truth_list)
                # Optional: Fallback date comparison for list elements if needed
            except Exception as e:
                # print(f"Error comparing list[category]: {e}")
                result = False
        elif semantic == "list[number]":
            try:
                def clean_and_convert(num_str):
                    cleaned = ''.join(c for c in num_str.strip(STRIP_CHARS) if c.isdigit() or c in ['.', '-'])
                    return round(float(cleaned), 2) if cleaned else None

                value_list_cleaned = [clean_and_convert(v) for v in value_str.split(',')]
                truth_list_cleaned = [clean_and_convert(t) for t in truth_str.split(',')]

                # Filter out None values that resulted from empty/invalid parts
                value_list_valid = [v for v in value_list_cleaned if v is not None]
                truth_list_valid = [t for t in truth_list_cleaned if t is not None]

                # Compare as sets for order independence
                result = set(value_list_valid) == set(truth_list_valid)
            except Exception as e:
                # print(f"Error comparing list[number]: {e}")
                result = False
        else:
            print(f"Warning: Unsupported semantic type encountered: '{semantic}'. Performing basic string comparison.")
            result = value_str.lower() == truth_str.lower()

    except Exception as e:
        print(f"Error during comparison (Type: '{semantic}', Value: '{value}', Truth: '{truth}'): {e}")
        result = False # Default to False on any unexpected error

    return result 
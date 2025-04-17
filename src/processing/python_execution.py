import pandas as pd
import numpy as np
import threading
import time
from typing import Tuple, Dict, Any

# Placeholder imports
from ..data.loader import load_table

def try_python_execution(code: str, dataset_name: str, is_sample: bool, timeout: int = 10) -> Tuple[Any, bool]: # Increased default timeout
    """Executes generated Python code in a separate thread with a timeout."""
    result = None
    success = False
    local_vars = {}
    exception_info = [None]

    def execute_code(code_to_run, df_loaded):
        nonlocal result, success, local_vars
        try:
            # Prepare execution context
            exec_globals = {'pd': pd, 'np': np, 'df': df_loaded}
            exec_locals = {}

            # Execute the code
            # Ensure the code assigns to 'result'
            exec(code_to_run, exec_globals, exec_locals)

            # Extract the result variable
            if 'result' in exec_locals:
                result = exec_locals['result']
                success = True
                # print(f"Execution successful. Result type: {type(result)}")
            else:
                 # Check if the last statement is an expression that evaluates to a value
                try:
                    # Try to evaluate the last line if it's an expression
                    last_line_result = eval(code_to_run.splitlines()[-1], exec_globals, exec_locals)
                    result = last_line_result
                    success = True
                    print("Execution successful. Inferred result from last line.")
                except Exception as eval_e:
                    print(f"Warning: 'result' variable not found in executed code, and last line evaluation failed: {eval_e}")
                    result = "__EXECUTION_FAILED__: 'result' variable not assigned and last line not evaluable."
                    success = False

        except Exception as e:
            print(f"Error during code execution: {e}")
            exception_info[0] = str(e)
            result = f"__EXECUTION_FAILED__: {str(e)}"
            success = False

    try:
        # Load the dataset required for the execution
        df = load_table(dataset_name, is_sample)
    except Exception as load_e:
         print(f"Error loading table '{dataset_name}' (sample={is_sample}): {load_e}")
         return f"__EXECUTION_FAILED__: Failed to load data - {str(load_e)}", False

    # Execute in a separate thread
    thread = threading.Thread(target=execute_code, args=(code, df))
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        # Thread is still running after timeout
        print(f"Execution timed out after {timeout} seconds.")
        # Attempt to forcefully stop the thread (Note: This is generally not recommended)
        # Python doesn't have a reliable way to kill a thread directly.
        # Depending on the task, other mechanisms like multiprocessing might be better.
        return "__EXECUTION_FAILED__: Timeout", False

    if exception_info[0]:
        # Exception occurred inside the thread
        return f"__EXECUTION_FAILED__: {exception_info[0]}", False

    # Return the captured result and success status
    # print(f"Final result before return: {result}, Success: {success}")
    return result, success

def format_result(result: Any) -> str:
    """Formats the execution result into a standardized string representation."""
    if isinstance(result, (pd.Series, pd.DataFrame)):
        # Handle pandas objects specifically if needed, e.g., convert to list or specific string format
        # For now, convert to string, but might need adjustment based on expected output format
        try:
             if isinstance(result, pd.Series):
                 # Convert Series to list, then string
                 return str(result.tolist())
             else: # DataFrame
                 # Convert DataFrame to string, might need a better format
                 return result.to_string()
        except Exception as format_e:
             print(f"Error formatting pandas object: {format_e}")
             return str(result) # Fallback to simple string conversion
    elif isinstance(result, (np.ndarray, list)):
        # Convert numpy arrays or lists to string
        # Ensure inner elements are also strings if necessary, especially for lists
        if isinstance(result, np.ndarray):
            result_list = result.tolist()
        else:
            result_list = result
        # Handle single-element lists
        if len(result_list) == 1:
             return str(result_list[0]).replace('"', '').replace("'", '') # Remove quotes for single items
        else:
             return str(result_list) # Keep list format for multiple items
    elif isinstance(result, str):
        # Clean up potential extra quotes from string results
        return result.replace('"", ').replace("'", '')
    else:
        # For other types (int, float, bool, None), convert directly to string
        return str(result) 
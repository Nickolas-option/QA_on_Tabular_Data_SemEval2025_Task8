import time
import re
from sqlalchemy import create_engine, types, MetaData, Table, text
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Dict, Tuple, Any
from fuzzysearch import find_near_matches

# Placeholder imports
from ..data.loader import load_table
from ..utils.prompt_utils import get_sql_prompt
from ..clients.openai_client import OpenAIClient # For type hint

# Global engine cache (consider thread-safety if using threads extensively here)
_engine = None

def get_db_connection():
    """Creates or returns a cached SQLite in-memory database engine."""
    global _engine
    if _engine is None:
        print("Creating new SQLite in-memory database engine.")
        _engine = create_engine('sqlite:///:memory:', echo=False)
        # Optional: Add listeners or configurations here if needed
    return _engine

def _load_df_to_sql(engine, df, table_name):
    """Loads a DataFrame into a SQL table, handling potential type issues."""
    try:
        # Define basic type mapping, ensure text for objects
        dtype_mapping = {col: types.UnicodeText for col, dtype in df.dtypes.items() if dtype == 'object'}
        df.to_sql(table_name, con=engine, index=False, if_exists='replace', dtype=dtype_mapping)
        # print(f"DataFrame successfully loaded into SQL table '{table_name}'.")
    except SQLAlchemyError as db_err:
        print(f"Database error loading DataFrame to SQL table '{table_name}': {db_err}")
        raise
    except Exception as e:
        print(f"Unexpected error loading DataFrame to SQL table '{table_name}': {e}")
        raise # Re-raise other errors

def _execute_sql_query(engine, sql_query: str) -> Tuple[Any, bool]:
    """Executes a SQL query and returns the result and success status."""
    try:
        with engine.connect() as connection:
            result_proxy = connection.execute(text(sql_query))
            rows = result_proxy.fetchall()
            # print(f"SQL executed successfully. Fetched {len(rows)} rows.")

            if not rows:
                return "__NO_RESULT__", False # Indicate no rows returned
            elif len(rows) == 1:
                # If single row, single column, return the value directly
                if len(rows[0]) == 1:
                    return rows[0][0], True
                else:
                    # Single row, multiple columns (shouldn't happen with prompt constraints but handle defensively)
                    return list(rows[0]), True
            else:
                # Multiple rows, assume single column as per prompt constraints
                return [row[0] for row in rows], True

    except SQLAlchemyError as db_err:
        print(f"Database error executing SQL: {db_err}\nQuery: {sql_query}")
        return f"__EXECUTION_FAILED__: DB Error - {str(db_err)}", False
    except Exception as e:
        print(f"Unexpected error executing SQL: {e}\nQuery: {sql_query}")
        return f"__EXECUTION_FAILED__: Unexpected Error - {str(e)}", False

def _clean_sql_response(sql_response: str) -> str:
    """Cleans the SQL query string received from the LLM."""
    # Find the start of the code block
    matches = find_near_matches('CODE:', sql_response, max_l_dist=2)
    if not matches:
         # Try finding markdown code block as fallback
        code_block_match = re.search(r'```(?:sql\n)?(.*?)```', sql_response, re.DOTALL)
        if code_block_match:
            sql_code = code_block_match.group(1).strip()
        else:
            print("Warning: Could not find 'CODE:' marker or SQL code block. Using raw response.")
            sql_code = sql_response # Use raw response as last resort
    else:
        # Extract code after the last 'CODE:' marker
        sql_code = sql_response[matches[-1].end:].strip()
        # Remove potential markdown fences if they exist after the marker
        if sql_code.startswith('```sql'):
            sql_code = sql_code[len('```sql'):].strip()
        if sql_code.startswith('```'):
             sql_code = sql_code[len('```'):].strip()
        if sql_code.endswith('```'):
            sql_code = sql_code[:-len('```')].strip()

    # Remove potential leading/trailing quotes or SQL keywords like 'sql'
    sql_code = sql_code.replace("sql", "", 1).strip().strip('`').strip()
    return sql_code

def sql_fallback(dataset_name: str, question: str, is_sample: bool, failed_solutions: List[Dict], ai_client: OpenAIClient, n_tries: int = 1, log_file: str = "sql_fallback_log.txt") -> List[Dict]:
    """Attempts to generate and execute SQL solutions as a fallback."""
    print(f"\n--- Initiating SQL Fallback for: {question[:50]}... ---")
    start_fallback_time = time.time()
    solutions = []
    errors = []

    try:
        engine = get_db_connection()
        df = load_table(dataset_name, is_sample)
        if df.empty:
             print("Skipping SQL fallback: DataFrame is empty.")
             return []

        # Use a unique, sanitized table name
        table_name = f"table_{dataset_name.replace('-','_').replace('.','_')}_{str(is_sample).lower()}"
        _load_df_to_sql(engine, df, table_name)

        # Reflect table metadata to get column names accurately
        metadata = MetaData()
        metadata.reflect(bind=engine)
        if table_name not in metadata.tables:
             print(f"Error: Table '{table_name}' not found after attempting to load.")
             return []
        temp_table = Table(table_name, metadata, autoload_with=engine)
        column_names = [column.name for column in temp_table.columns]

    except Exception as setup_e:
        print(f"Error during SQL fallback setup: {setup_e}")
        return [] # Cannot proceed if setup fails

    # --- SQL Generation and Execution Loop ---
    current_prompt = get_sql_prompt(failed_solutions, question, column_names, df, ai_client).replace("temp_table", f'"{table_name}"') # Ensure table name is quoted

    for attempt in range(n_tries):
        print(f"SQL Generation Attempt {attempt + 1}/{n_tries}")
        sql_query = ""
        try:
            if attempt > 0:
                error_msg = "; ".join(errors)
                retry_prompt = f"Previous SQL attempt failed with error: {error_msg}. Please provide a corrected SQL query based on the original instructions.\nOriginal Task Context:\n{current_prompt}"
                sql_response = ai_client.generate_response(retry_prompt)
            else:
                sql_response = ai_client.generate_response(current_prompt)

            sql_query = _clean_sql_response(sql_response)

            if not sql_query:
                 print("Warning: Empty SQL query generated or cleaned. Skipping execution.")
                 errors.append("Empty SQL query generated.")
                 solutions.append({"code": "", "result": "__GENERATION_FAILED__", "success": False})
                 continue # Skip to next attempt if query is empty

            # Log the attempt
            with open(log_file, "a", encoding='utf-8') as f:
                f.write(f"-- Attempt {attempt + 1} --\n")
                f.write(f"Question: {question}\n")
                f.write(f"Columns: {column_names}\n")
                f.write(f"Generated SQL:\n{sql_query}\n")

            # Execute the cleaned SQL query
            result, success = _execute_sql_query(engine, sql_query)
            solutions.append({"code": sql_query, "result": str(result), "success": success})

            if success:
                print(f"SQL Attempt {attempt + 1} succeeded.")
                # Optionally break early on success, or run all n_tries
                # break
            else:
                print(f"SQL Attempt {attempt + 1} failed. Result: {result}")
                if isinstance(result, str) and result.startswith("__EXECUTION_FAILED__"):
                    errors.append(result) # Log specific execution error
                else:
                    errors.append(f"Execution failed with result: {result}")

        except Exception as gen_exec_e:
            print(f"Error during SQL generation or execution attempt {attempt + 1}: {gen_exec_e}")
            errors.append(f"Attempt {attempt + 1} Error: {str(gen_exec_e)}")
            solutions.append({"code": sql_query or "__QUERY_GENERATION_FAILED__", "result": f"__EXECUTION_FAILED__: {str(gen_exec_e)}", "success": False})

    print(f"--- SQL Fallback finished in {time.time() - start_fallback_time:.2f} seconds. {sum(s['success'] for s in solutions)} successful attempts. ---")
    return solutions 
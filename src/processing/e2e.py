import re
import pandas as pd
from typing import List, Dict
from fuzzysearch import find_near_matches

# Placeholder imports
from ..data.loader import load_table
from ..clients.openai_client import OpenAIClient # For type hint
from ..utils.embedding_utils import get_relevant_columns_only # Use this utility

def dataset_to_text(dataset_name: str, question: str, ai_client: OpenAIClient, is_sample: bool) -> str:
    """
    Loads a dataset, selects relevant columns, and converts it into a markdown representation.
    """
    print(f"Converting dataset '{dataset_name}' (sample={is_sample}) to text for E2E...")
    try:
        # Load the full table first
        full_df = load_table(dataset_name, is_sample)
        if full_df.empty:
            print("Warning: Loaded DataFrame is empty.")
            return "(Dataset is empty)"

        # Get only relevant columns using the utility function
        # This requires the ai_client to be passed
        relevant_df = get_relevant_columns_only(full_df, question, ai_client)

        if relevant_df.empty:
            print("Warning: No relevant columns identified by LLM for E2E. Using full DataFrame for markdown.")
            # Fallback to using the full DataFrame if no relevant columns are found
            target_df = full_df
        else:
            target_df = relevant_df

        # Convert the selected DataFrame to markdown
        markdown_text = target_df.to_markdown(index=False) # Exclude index for cleaner representation

        # Ensure the markdown text is UTF-8 encoded
        markdown_text = markdown_text.encode('utf-8', errors='ignore').decode('utf-8')

        # Optional: Truncate if too long (though E2E might need full context)
        # truncate_limit = 10000 # Example limit
        # if len(markdown_text) > truncate_limit:
        #     markdown_text = markdown_text[:truncate_limit] + "\n... (truncated)"

        # print(f"Dataset converted to markdown (length: {len(markdown_text)} chars).")
        # with open('dataset_markdown_log.txt', 'a', encoding='utf-8') as f:
        #     f.write(f"--- {dataset_name} ---\n{markdown_text}\n---\n")
        return markdown_text

    except Exception as e:
        print(f"Error during dataset_to_text conversion: {e}")
        return f"(Error converting dataset to text: {e})"

def e2e_response(question: str, dataset_name: str, is_sample: bool, ai_client: OpenAIClient) -> List[Dict]:
    """Generates an end-to-end response directly from the dataset text."""
    print(f"\n--- Generating E2E Response for: {question[:50]}... ---")
    start_e2e_time = time.time()

    # Get dataset as markdown text
    dataset_text = dataset_to_text(dataset_name, question, ai_client, is_sample)

    if dataset_text.startswith("(Error") or dataset_text == "(Dataset is empty)":
        print(f"Skipping E2E generation due to dataset text issue: {dataset_text}")
        return []

    prompt = f'''
Question: {question}
Dataset:
```markdown
{dataset_text}
```

Analyze the data provided above. Provide your final answer to the question based *only* on this data.
Your answer MUST be in one of the following exact formats:
1. Boolean: True / False
2. List: e.g., ['apple', 'banana', 'cherry'] (use single quotes for strings within the list)
3. Number: e.g., 42 / 3.14
4. String: e.g., 'Spanish' (use single quotes)

First, provide concise step-by-step reasoning based *only* on the provided data. Do not infer or use outside knowledge.
Then, provide the final answer prefixed *exactly* with "Final Answer:", followed by the answer in one of the required formats.
Your response must end immediately after the final answer.

Reasoning:
[Your reasoning steps here]

Final Answer: [Your answer here in one of the required formats]
'''

    try:
        # Use a capable model for reasoning and direct answering
        # model_name = "minimax/minimax-01" # From original notebook
        # model_name = "google/gemini-pro" # Alternative strong model
        model_name = "meta-llama/llama-3.3-70b-instruct" # Another strong model
        print(f"Sending E2E prompt to model: {model_name}")
        raw_response = ai_client.generate_response(prompt, model=model_name)
        print(f"E2E response received from LLM.")

        # Find the last occurrence of "Final Answer:"
        matches = find_near_matches('Final Answer:', raw_response, max_l_dist=1)
        if not matches:
            print("Warning: Could not find 'Final Answer:' marker in E2E response.")
            print(f"Raw E2E Response:\n{raw_response}")
            return [{
                "code": "End-to-End model (no code visible)",
                "result": "__E2E_RESPONSE_PARSE_FAILED__",
                "success": False # Mark as failed if parsing fails
            }]

        # Extract text after the last marker
        final_answer_text = raw_response[matches[-1].end:].strip()

        # Basic cleaning: remove potential markdown, extra quotes sometimes added by LLM
        cleaned_answer = final_answer_text.replace('**', '').replace('*', '').replace('`', '').replace('#', '').replace('_', '').strip()

        # Attempt to further refine based on expected formats (Boolean, List, Number, String)
        # This is tricky as LLM output varies. We try to match the most likely format.

        # Check for Boolean explicitly
        if cleaned_answer.lower() == 'true':
            final_result = 'True'
        elif cleaned_answer.lower() == 'false':
            final_result = 'False'
        # Check for List (heuristic: starts with [ and ends with ])
        elif cleaned_answer.startswith('[') and cleaned_answer.endswith(']'):
            # Keep as string list representation for now, evaluation logic handles parsing
            final_result = cleaned_answer
        # Check for Number (heuristic: can be converted to float/int)
        else:
            try:
                # Try converting to float, then int if it's whole
                num_val = float(cleaned_answer)
                if num_val.is_integer():
                     final_result = str(int(num_val))
                else:
                     final_result = str(num_val)
            except ValueError:
                # If not boolean, list, or number, treat as String
                # Remove potential enclosing quotes if they exist
                if (cleaned_answer.startswith("'") and cleaned_answer.endswith("'")) or \
                   (cleaned_answer.startswith('"') and cleaned_answer.endswith('"')):
                   final_result = cleaned_answer[1:-1]
                else:
                   final_result = cleaned_answer

        print(f"E2E processing finished in {time.time() - start_e2e_time:.2f} seconds.")
        return [{
            "code": "End-to-End model (no code visible)",
            "result": final_result,
            "success": True
        }]

    except Exception as e:
        print(f"Error during E2E response generation or processing: {e}")
        print(f"Raw E2E Response (if available):\n{raw_response if 'raw_response' in locals() else 'N/A'}")
        return [{
            "code": "End-to-End model (no code visible)",
            "result": f"__E2E_GENERATION_ERROR__: {str(e)}",
            "success": False
        }] 
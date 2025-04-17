import re
from fuzzysearch import find_near_matches

def clean_code_response(code_response: str) -> str:
    """Cleans the code response string received from the LLM."""
    matches = find_near_matches('Code:', code_response, max_l_dist=2)
    if matches:
        code_response = code_response[matches[-1].end:]
    else:
        # Attempt to find code blocks or common patterns if 'Code:' is missing
        code_block_match = re.search(r'```(?:python\n)?(.*?)```', code_response, re.DOTALL)
        if code_block_match:
            code_response = code_block_match.group(1).strip()
        else:
            # Fallback or raise error if no code found
            print("Warning: 'Code:' marker not found and no python code block detected. Trying to extract last line as code.")
            # As a last resort, try taking the last non-empty line
            lines = [line for line in code_response.splitlines() if line.strip()]
            if lines:
                code_response = lines[-1]
            else:
                 raise ValueError("Expected 'Code:' or a code block in the response but none found.")


    # General cleaning
    code_response = code_response.replace("*", "").replace("`", "").replace("python", "").replace('return', '').strip()

    # Ensure it's likely a single line of code, take the last line if multiple exist
    lines = [line for line in code_response.splitlines() if line.strip()]
    if lines:
        code_response = lines[-1]
    else:
        code_response = "" # Handle cases where cleaning results in empty string

    # Remove specific DataFrame creation pattern
    code_response = re.sub(r"df\s*=\s*pd\.DataFrame\(\[\{.*?\}\]\);?", "", code_response, flags=re.DOTALL).strip()
    # Remove potential leading semicolons
    code_response = code_response.lstrip(';').strip()
    return code_response 
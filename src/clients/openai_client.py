import openai
import os
from dotenv import load_dotenv
import time

class OpenAIClient:
    """A client to interact with the OpenAI API, configured for OpenRouter."""
    def __init__(self):
        load_dotenv() # Load environment variables from .env file
        self.api_key = os.getenv('OPENROUTER_API_KEY') # Use a generic name or make configurable
        if not self.api_key:
             # Fallback to another key if the first one isn't set
             self.api_key = os.getenv('OPENROUTER_API_KEY2')
        if not self.api_key:
            raise ValueError("API key not found. Please set OPENROUTER_API_KEY or OPENROUTER_API_KEY2 in your .env file.")

        self.base_url = os.getenv('OPENROUTER_BASE_URL', "https://openrouter.ai/api/v1") # Default if not set

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        print(f"OpenAIClient initialized with base URL: {self.base_url}")

    def generate_response(self, prompt: str, max_tokens=3000, retries=3, model="meta-llama/llama-3.3-70b-instruct") -> str:
        """Generates a response from the specified model with retry logic."""
        sleep_durations = [15, 30, 60]  # Sleep durations in seconds for retries

        for attempt in range(retries):
            try:
                print(f"Attempt {attempt + 1} to generate response with model {model}...")
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a best in class instruction following assistant. You excel in tasks with data. You reason very thorougly and step-by-step."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0, # For deterministic results
                    max_tokens=max_tokens
                )
                # print(response) # Uncomment for debugging

                if response.choices:
                    content = response.choices[0].message.content.strip()
                    # print(f"{model} response received.") # Uncomment for debugging
                    return content
                else:
                    # This case should ideally not happen with valid API calls
                    # but good to handle defensively.
                    print("Warning: No response choices received from API.")
                    # Consider specific error handling or raising a custom exception
                    raise ValueError("No response choices available from the API.")

            except openai.RateLimitError as e:
                 print(f"Attempt {attempt + 1} failed: Rate limit error: {e}")
                 if attempt < len(sleep_durations):
                     sleep_time = sleep_durations[attempt]
                     print(f"Rate limited. Sleeping for {sleep_time} seconds before retrying...")
                     time.sleep(sleep_time)
                 else:
                     print("Max retries reached after rate limit error.")
                     return f"__CODE_GEN_ERROR__: Rate limit exceeded after {retries} attempts."
            except openai.APIError as e:
                print(f"Attempt {attempt + 1} failed: OpenAI API error: {e}")
                # General API errors might be retryable
                if attempt < len(sleep_durations):
                     sleep_time = sleep_durations[attempt]
                     print(f"API error encountered. Sleeping for {sleep_time} seconds before retrying...")
                     time.sleep(sleep_time)
                else:
                     print("Max retries reached after API error.")
                     return f"__CODE_GEN_ERROR__: API error after {retries} attempts: {str(e)}"
            except Exception as e:
                # Catch other potential exceptions (network issues, etc.)
                print(f"Attempt {attempt + 1} failed with unexpected error: {e}")
                # Decide if retrying makes sense for this type of error
                if attempt < len(sleep_durations):
                    sleep_time = sleep_durations[attempt]
                    print(f"Unexpected error. Sleeping for {sleep_time} seconds before retrying...")
                    time.sleep(sleep_time)
                else:
                    print("Max retries reached after unexpected error.")
                    # Return a generic error message or re-raise the exception
                    return f"__CODE_GEN_ERROR__: An unexpected error occurred after {retries} attempts: {str(e)}"

        # This point should only be reached if all retries fail
        return f"__CODE_GEN_ERROR__: Failed to generate response after {retries} attempts." 
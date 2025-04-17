import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from typing import List

# Placeholder imports - these will be resolved later
from ..data.loader import load_table
# from ..clients.openai_client import OpenAIClient # Avoid circular dependency for type hint

# Initialize model globally to avoid reloading
_model = None

def get_model():
    """Initializes and returns the SentenceTransformer model."""
    global _model
    if _model is None:
        # Consider making the model name configurable
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def get_relevant_columns_only(df: pd.DataFrame, question: str, ai_client) -> pd.DataFrame:
    """Uses an LLM to determine and return only the relevant columns for a question."""
    # ai_client is an instance of OpenAIClient
    prompt = f"Given the question: '{question}', which columns from the dataset are necessary to answer it? The dataset contains the following columns: {', '.join(df.columns)}. Please provide extensive reasoning and only then the column names as a comma-separated list started with [ and ended with ]."
    try:
        # Consider making the model name configurable
        columns_text = ai_client.generate_response(prompt, model="meta-llama/llama-3.2-3b-instruct").strip()
        columns_match = re.search(r'\[(.*?)\]', columns_text)
        if columns_match:
            columns_str = columns_match.group(1)
            columns_list = [col.strip().strip("'\"` ") for col in columns_str.split(',') if col.strip()]
            # Validate columns exist in df
            valid_columns = [col for col in columns_list if col in df.columns]
            if not valid_columns:
                 print(f"Warning: LLM identified columns not in DataFrame: {columns_list}. Using all columns.")
                 return df # Return original df if no valid columns identified
            return df[valid_columns]
        else:
            print("Warning: Could not parse columns from LLM response. Using all columns.")
            return df # Return original df if parsing fails
    except Exception as e:
        print(f"Error getting relevant columns from LLM: {e}. Using all columns.")
        return df # Return original df on error

def get_relevant_rows_by_cosine_similarity(df: pd.DataFrame, question: str, ai_client, top_n: int = 10) -> pd.DataFrame:
    """Retrieves the top_n most relevant rows based on cosine similarity to the question."""
    # ai_client is an instance of OpenAIClient
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = get_model().to(device)

    try:
        # Get relevant columns first
        relevant_df = get_relevant_columns_only(df, question, ai_client=ai_client)

        if relevant_df.empty:
            print("Warning: No relevant columns identified or DataFrame is empty. Returning top 3 rows.")
            return df.head(3)

        # Efficiently create text representation for embedding
        # Handle potential mixed types by converting all to string
        text_data = relevant_df.astype(str).agg(' '.join, axis=1).tolist()

        if not text_data:
             print("Warning: No text data generated from relevant columns. Returning top 3 rows.")
             return df.head(3)

        # Encode data and question
        embeddings = model.encode(
            text_data,
            batch_size=256, # Consider making batch size configurable
            show_progress_bar=False,
            convert_to_tensor=True,
            device=device
        )
        question_embedding = model.encode(
            [question],
            show_progress_bar=False,
            convert_to_tensor=True,
            device=device
        )[0]

        # Calculate cosine similarities efficiently
        # Move embeddings to CPU for numpy operations if they were on GPU
        if embeddings.device != torch.device('cpu'):
            embeddings = embeddings.cpu()
        if question_embedding.device != torch.device('cpu'):
            question_embedding = question_embedding.cpu()

        # Normalize embeddings
        embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        question_embedding_norm = torch.nn.functional.normalize(question_embedding.unsqueeze(0), p=2, dim=1)

        # Cosine similarity
        similarities = torch.mm(embeddings_norm, question_embedding_norm.transpose(0, 1)).squeeze()

        # Find top N indices efficiently
        # Ensure top_n is not greater than the number of rows
        actual_top_n = min(top_n, len(similarities))
        if actual_top_n <= 0:
            return df.head(3) # Return head if no similarities or top_n is non-positive

        # Use torch.topk for efficiency
        _, top_indices = torch.topk(similarities, actual_top_n)

        # Return the corresponding rows from the *original* DataFrame
        return df.iloc[top_indices.cpu().numpy()]

    except Exception as e:
        print(f"Error in similarity calculation: {e}. Returning top 3 rows.")
        # Fallback to returning the head of the original DataFrame
        return df.head(3) 
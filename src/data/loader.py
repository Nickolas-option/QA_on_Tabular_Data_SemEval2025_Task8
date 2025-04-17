import os
import pandas as pd
from datasets import load_dataset, Dataset
from functools import lru_cache

# Import utility function
from ..utils.text_utils import replace_emojis_with_unique_symbols

# Cache for preloaded datasets to avoid redundant loading
preloaded_datasets = {}

def load_data(name: str = "semeval", split: str = "dev", data_dir: str = "./data") -> Dataset:
    """Load dataset from local parquet file or download if not available."""
    os.makedirs(data_dir, exist_ok=True) # Ensure data directory exists
    local_path = os.path.join(data_dir, f"{name}_{split}.parquet")

    if os.path.exists(local_path):
        print(f"Loading dataset from local path: {local_path}")
        return Dataset.from_parquet(local_path)

    print(f"Local dataset not found. Downloading {name}/{split}...")
    # Ensure the dataset exists before trying to download
    try:
        dataset = load_dataset("cardiffnlp/databench", name=name, split=split)
        dataset.to_parquet(local_path)
        print(f"Dataset saved locally to: {local_path}")
        return dataset
    except Exception as e:
        print(f"Failed to load or save dataset {name}/{split}: {e}")
        raise # Re-raise the exception after logging

@lru_cache(maxsize=None) # Cache results of load_table
def load_table(dataset_name: str, is_sample: bool = False, data_dir: str = "./data") -> pd.DataFrame:
    """Load table from local parquet file or remote source if not cached."""
    os.makedirs(data_dir, exist_ok=True) # Ensure data directory exists
    file_type = "sample" if is_sample else "all"
    key = f"{dataset_name}_{file_type}"
    local_path = os.path.join(data_dir, f"{dataset_name}_{file_type}.parquet")

    if key not in preloaded_datasets:
        if os.path.exists(local_path):
            print(f"Loading table from local path: {local_path}")
            df = pd.read_parquet(local_path)
        else:
            remote_path = f"hf://datasets/cardiffnlp/databench/data/{dataset_name}/{file_type}.parquet"
            print(f"Local table not found. Downloading from: {remote_path}")
            try:
                df = pd.read_parquet(remote_path)
                # Save the downloaded file locally
                df.to_parquet(local_path)
                print(f"Table saved locally to: {local_path}")
            except Exception as e:
                print(f"Failed to load or save table {key}: {e}")
                # Return an empty DataFrame or raise an error depending on desired behavior
                raise

        # Process columns only once after loading
        print(f"Processing columns for {key}...")
        df.columns = df.columns.str.replace(r'<gx:.*?>', '', regex=True)
        df = replace_emojis_with_unique_symbols(df) # Use imported function
        df.columns = [f"{col}" for col in df.columns] # Simplified column naming
        # df.columns = [f"{col}_{i}" for i, col in enumerate(df.columns)] # Alternative if needed
        df._processed = True # Mark as processed
        preloaded_datasets[key] = df
    else:
        print(f"Loading table {key} from cache.")
        df = preloaded_datasets[key]

    return df.copy() # Return a copy to prevent modification of cached data

def load_sample(name: str = "qa", data_dir: str = "./data") -> pd.DataFrame:
    """Load sample data from local parquet file or remote source if not available."""
    # This function seems redundant given load_table with is_sample=True,
    # but keeping it for compatibility with the original notebook structure.
    # It might be worth consolidating later.
    os.makedirs(data_dir, exist_ok=True)
    local_path = os.path.join(data_dir, f"{name}_sample.parquet")

    if os.path.exists(local_path):
        print(f"Loading sample from local path: {local_path}")
        return pd.read_parquet(local_path)

    remote_path = f"hf://datasets/cardiffnlp/databench/data/{name}/sample.parquet"
    print(f"Local sample not found. Downloading from: {remote_path}")
    try:
        df = pd.read_parquet(remote_path)
        # Process columns similar to load_table for consistency
        df.columns = df.columns.str.replace(r'<gx:.*?>', '', regex=True)
        df = replace_emojis_with_unique_symbols(df)
        df.columns = [f"{col}" for col in df.columns]
        df.to_parquet(local_path)
        print(f"Sample saved locally to: {local_path}")
        return df
    except Exception as e:
        print(f"Failed to load or save sample {name}: {e}")
        raise 
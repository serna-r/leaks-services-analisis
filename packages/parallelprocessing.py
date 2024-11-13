import concurrent.futures
import pandas as pd
from functools import partial
from packages.stats import apply_all_in_one, apply_one

# Number of chunks or chunk size
chunk_size = 1000  # Adjust chunk size based on your needs


# Function to split DataFrame into chunks
def split_dataframe(df, chunk_size):
    chunks = [df[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]
    return chunks

# Function to process a chunk of the DataFrame
def process_chunk_all(chunk):
    chunk['mask'], chunk['total'], chunk['length'], chunk['simple_entropy'], chunk['shannon_entropy'], chunk['common'], chunk['score'], chunk['guesses'] = zip(*chunk['Password'].map(apply_all_in_one))
    return chunk

# Function to process a chunk of the DataFrame
def process_chunk_one(chunk, metric):
    chunk[metric] = chunk['Password'].apply(lambda x: apply_one(x, metric))
    return chunk

# Function to process one stat in a parallel way
def parallel_proc_one(df, metric):
    # Split DataFrame into chunks
    chunks = split_dataframe(df, chunk_size)

    # Use ThreadPoolExecutor or ProcessPoolExecutor for parallel processing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Use partial to fix the metric parameter
        process_chunk_with_metric = partial(process_chunk_one, metric=metric)
        # Map the process_chunk function to each chunk
        results = list(executor.map(process_chunk_with_metric, chunks))

    # Combine all the processed chunks back into a single DataFrame
    df_processed = pd.concat(results, ignore_index=True)

    return df_processed


# Function to process all stats in a parallel way
def parallel_proc_all(df):
    # Number of chunks or chunk size
    chunk_size = 1000  # Adjust chunk size based on your needs

    # Split DataFrame into chunks
    chunks = split_dataframe(df, chunk_size)

    # Use ThreadPoolExecutor or ProcessPoolExecutor for parallel processing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Map the process_chunk function to each chunk
        results = list(executor.map(process_chunk_all, chunks))

    # Combine all the processed chunks back into a single DataFrame
    df_processed = pd.concat(results, ignore_index=True)

    return df_processed
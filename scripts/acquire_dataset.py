# scripts/acquire_dataset.py


import pandas as pd
import os
import logging
import sys
import argparse # What it does: Python's standard library for parsing command-line arguments.


# --- Dependency Check ---
try:
    import fsspec
    import huggingface_hub
except ImportError:
    logging.error("Libraries fsspec and huggingface_hub are required. Please run: pip install fsspec huggingface_hub")
    sys.exit(1)


# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def setup_arg_parser():
    """
    What it does: Sets up the command-line argument parser.
    Why it's here (The Rationale): Makes the script configurable from the outside,
    allowing us to request either the full dataset or a sample of a specific size
    without modifying the code. This is essential for a hybrid dev/prod workflow.
    """
    parser = argparse.ArgumentParser(description="Download and process fraud dataset from Hugging Face.")
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None, # By default, use the full dataset.
        help="Number of rows to sample randomly. If not provided, the full dataset will be saved."
    )
    return parser


def acquire_dataset(sample_size):
    """
    Downloads dataset from Hugging Face, optionally samples it, and saves it locally.
    Args:
        sample_size (int, optional): The number of rows to sample. Defaults to None (full dataset).
    """
    # --- Configuration ---
    hf_csv_url = "hf://datasets/MatrixIA/FraudData/FraudData.csv"
    output_dir = "data/raw_data/"
    
    # --- Dynamic Filename ---
    # What it does: Creates a different filename depending on whether we're saving a sample or the full dataset.
    # Why it's here (The Rationale): Prevents accidentally overwriting the full dataset with a sample,
    # and makes it clear which file is which.
    if sample_size:
        output_filename = f"sampled_fraud_data_{sample_size}.csv"
    else:
        output_filename = "full_fraud_data.csv"
    
    output_path = os.path.join(output_dir, output_filename)
    
    logging.info("Starting dataset acquisition process...")
    os.makedirs(output_dir, exist_ok=True)


    try:
        logging.info(f"Attempting to load FULL dataset from: {hf_csv_url}")
        logging.warning("This may take several minutes and consume significant RAM (est. 2-4 GB).")


        df_full = pd.read_csv(hf_csv_url)
        logging.info(f"Successfully loaded full dataset. Shape: {df_full.shape}")


        # --- Conditional Sampling Logic ---
        if sample_size:
            logging.info(f"Creating a random sample of {sample_size:,} rows...")
            # What it does: Checks if the requested sample size is valid.
            # Why it's here (The Rationale): Robust error handling for user input.
            if len(df_full) < sample_size:
                logging.warning(f"Full dataset size ({len(df_full)}) is smaller than requested sample size ({sample_size}). Using full dataset for sampling.")
                df_to_save = df_full.copy()
            else:
                # Implementation Detail: `random_state` ensures that if you ask for the same sample size
                # multiple times, you get the exact same random sample. This is key for reproducibility.
                df_to_save = df_full.sample(n=sample_size, random_state=50).copy()
        else:
            logging.info("No sample size provided. Preparing to save the full dataset.")
            df_to_save = df_full


        logging.info(f"Saving data to '{output_path}'. Final shape: {df_to_save.shape}")
        df_to_save.to_csv(output_path, index=False)
        logging.info(f"Successfully saved dataset to {output_path}")


    except Exception as e:
        logging.error(f"Failed during dataset acquisition. Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()
    acquire_dataset(sample_size=args.sample_size)
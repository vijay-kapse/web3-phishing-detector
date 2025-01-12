import pandas as pd
import numpy as np
from sklearn.utils import resample
import os
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('preprocessing.log')
    ]
)

def load_datasets(data_dir):
    """
    Load all CSV datasets from the data directory.
    
    Args:
        data_dir (str): Path to directory containing CSV files
        
    Returns:
        list: List of pandas DataFrames
    """
    datasets = []
    logging.info("Scanning directory: %s", data_dir)
    
    try:
        files = os.listdir(data_dir)
        csv_files = [f for f in files if f.endswith('.csv')]
        logging.info("Found %d CSV files: %s", len(csv_files), csv_files)
        
        for file in csv_files:
            file_path = os.path.join(data_dir, file)
            logging.info("Loading dataset: %s", file)
            try:
                df = pd.read_csv(file_path)
                logging.info("Successfully loaded %s: %d rows, %d columns", 
                           file, len(df), len(df.columns))
                logging.info("Columns: %s", df.columns.tolist())
                datasets.append((file, df))
            except Exception as e:
                logging.error("Error loading %s: %s", file, str(e))
                continue
    except Exception as e:
        logging.error("Error accessing directory %s: %s", data_dir, str(e))
        raise
    
    return datasets

def preprocess_text(text):
    """
    Clean and preprocess text data.
    
    Args:
        text: Input text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    if pd.isnull(text):
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Basic cleaning
    text = text.lower()
    text = text.strip()
    
    return text

def combine_text_fields(row, text_cols):
    """Combine multiple text fields into a single text."""
    texts = []
    for col in text_cols:
        if col in row and pd.notna(row[col]):
            texts.append(str(row[col]))
    return " ".join(texts)

def is_web3_related(text):
    """
    Check if text is related to Web3/crypto.
    
    Args:
        text (str): Input text to check
        
    Returns:
        bool: True if text contains Web3-related keywords
    """
    web3_keywords = {
        'crypto', 'bitcoin', 'ethereum', 'wallet', 'blockchain', 'nft',
        'defi', 'token', 'web3', 'metamask', 'mining', 'coin', 'dao',
        'smart contract', 'dex', 'exchange', 'airdrop', 'btc', 'eth',
        'binance', 'coinbase', 'opensea', 'uniswap', 'ledger', 'trezor',
        'seed phrase', 'private key', 'public key', 'gas fee', 'gwei',
        'metamask', 'trustwallet', 'blockchain', 'cryptocurrency'
    }
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in web3_keywords)

def process_dataset(file_name, df):
    """Process a single dataset to extract text and label."""
    try:
        # Define possible column names for text and labels
        text_columns = {
            'text', 'content', 'message', 'email', 'body', 'subject',
            'text_combined'
        }
        label_columns = {
            'label', 'class', 'spam', 'phishing', 'is_phishing',
            'is_spam', 'target'
        }
        
        # Find text columns
        available_text_cols = [col for col in df.columns if any(
            text_col in col.lower() for text_col in text_columns
        )]
        
        if not available_text_cols:
            raise ValueError(f"No text columns found in {file_name}")
        
        logging.info("Found text columns: %s", available_text_cols)
        
        # Find label column
        label_col = next(col for col in df.columns if any(
            label_col in col.lower() for label_col in label_columns
        ))
        logging.info("Found label column: %s", label_col)
        
        # Combine text fields
        df['text'] = df.apply(
            lambda row: combine_text_fields(row, available_text_cols),
            axis=1
        )
        
        # Clean text
        df['text'] = df['text'].apply(preprocess_text)
        
        # Normalize labels to 0/1
        df['label'] = df[label_col].map(lambda x: 1 if x in [1, '1', 'spam', 'phishing', True] else 0)
        
        # Select final columns
        result_df = df[['text', 'label']].copy()
        logging.info("Processed %d rows from %s", len(result_df), file_name)
        
        return result_df
    
    except Exception as e:
        logging.error("Error processing dataset %s: %s", file_name, str(e))
        return None

def create_dataset(data_dir, output_path):
    """
    Create a balanced dataset of Web3-related phishing and legitimate messages.
    
    Args:
        data_dir (str): Input directory containing CSV files
        output_path (str): Path to save the processed dataset
    """
    logging.info("Starting dataset creation process")
    logging.info("Input directory: %s", data_dir)
    logging.info("Output path: %s", output_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load datasets
    logging.info("Loading datasets...")
    datasets = load_datasets(data_dir)
    logging.info("Loaded %d datasets", len(datasets))
    
    # Process each dataset
    all_data = []
    for i, (file_name, df) in enumerate(datasets, 1):
        logging.info("Processing dataset %d/%d: %s", i, len(datasets), file_name)
        processed_df = process_dataset(file_name, df)
        if processed_df is not None:
            all_data.append(processed_df)
    
    # Combine all processed datasets
    if not all_data:
        raise ValueError("No datasets were successfully processed")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    logging.info("Combined dataset size: %d rows", len(combined_df))
    
    # Filter Web3-related content
    logging.info("Filtering Web3-related content...")
    web3_mask = combined_df['text'].apply(is_web3_related)
    web3_df = combined_df[web3_mask].copy()
    logging.info("Found %d Web3-related messages", len(web3_df))
    
    # Balance the dataset
    logging.info("Balancing dataset...")
    phishing = web3_df[web3_df['label'] == 1]
    legitimate = web3_df[web3_df['label'] == 0]
    
    logging.info("Initial class distribution:")
    logging.info("Phishing: %d samples", len(phishing))
    logging.info("Legitimate: %d samples", len(legitimate))
    
    # Use the smaller class size for balancing
    n_samples = min(len(phishing), len(legitimate))
    logging.info("Balancing to %d samples per class", n_samples)
    
    # Downsample the majority class
    if len(phishing) > n_samples:
        phishing = resample(phishing, n_samples=n_samples, random_state=42)
    if len(legitimate) > n_samples:
        legitimate = resample(legitimate, n_samples=n_samples, random_state=42)
    
    # Combine and shuffle
    final_df = pd.concat([phishing, legitimate])
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the processed dataset
    logging.info("Saving %d samples to %s...", len(final_df), output_path)
    try:
        final_df.to_csv(output_path, index=False)
        logging.info("Dataset successfully saved!")
        
        # Verify the saved file
        saved_df = pd.read_csv(output_path)
        logging.info("Verification: Saved file contains %d rows and %d columns",
                    len(saved_df), len(saved_df.columns))
        
        # Log some statistics
        logging.info("Dataset statistics:")
        logging.info("Average text length: %.2f characters", 
                    saved_df['text'].str.len().mean())
        logging.info("Class distribution:")
        logging.info(saved_df['label'].value_counts().to_dict())
        
        # Log sample texts
        logging.info("Sample phishing text:")
        logging.info(saved_df[saved_df['label'] == 1]['text'].iloc[0][:200] + "...")
        logging.info("Sample legitimate text:")
        logging.info(saved_df[saved_df['label'] == 0]['text'].iloc[0][:200] + "...")
        
    except Exception as e:
        logging.error("Error saving dataset: %s", str(e))
        raise

if __name__ == "__main__":
    try:
        create_dataset(
            data_dir="data",
            output_path="data/phishing_dataset.csv"
        )
    except Exception as e:
        logging.error("Fatal error: %s", str(e))
        sys.exit(1) 
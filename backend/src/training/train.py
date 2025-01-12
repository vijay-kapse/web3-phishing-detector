import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import wandb
from tqdm import tqdm
import os
import logging
import sys
from typing import Dict, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AdamW
from torch import nn
from pathlib import Path

from ..models.phishing_detector import PhishingDetector
from ..data.dataset import PhishingDataset

def calculate_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    predictions_binary = predictions >= 0.5
    return {
        'accuracy': accuracy_score(labels, predictions_binary),
        'precision': precision_score(labels, predictions_binary),
        'recall': recall_score(labels, predictions_binary),
        'f1': f1_score(labels, predictions_binary)
    }

def train_epoch(model, train_data, dataset, device, optimizer, criterion):
    model.train()
    total_loss = 0
    progress_bar = tqdm(range(len(train_data)))
    
    for batch_idx, batch in enumerate(train_data):
        try:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].float().to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.update(1)
            
        except Exception as e:
            logging.error(f"Error processing batch: {str(e)}")
            continue
            
    return total_loss / len(train_data)

def evaluate(model, val_data, dataset, device, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_data:
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].float().to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.squeeze(), labels)
                
                total_loss += loss.item()
                
                preds = (outputs.squeeze() > 0.5).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                
            except Exception as e:
                logging.error(f"Error processing row during evaluation: {str(e)}")
                continue
    
    if len(all_preds) == 0:
        raise ValueError("No valid predictions during evaluation")
        
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds)
    }
    
    return total_loss / len(val_data), metrics

def train(config):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load tokenizer and create datasets
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    # Load the dataset
    df = pd.read_csv(config['data_path'])
    
    # Take subset of data for quick testing
    df = df.head(500)  # Only use first 500 samples for testing
    
    # Create train/val/test splits
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Create datasets
    train_dataset = PhishingDataset(train_df, tokenizer)
    val_dataset = PhishingDataset(val_df, tokenizer)
    test_dataset = PhishingDataset(test_df, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    # Initialize model
    model = PhishingDetector(config['model_name']).to(device)
    
    # Initialize optimizer and loss
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = nn.BCELoss()
    
    # Initialize early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(config['epochs']):
        logging.info(f"Epoch {epoch+1}/{config['epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, train_dataset, device, optimizer, criterion)
        logging.info(f"Training loss: {train_loss:.4f}")
        
        # Evaluate
        val_loss, val_metrics = evaluate(model, val_loader, val_dataset, device, criterion)
        logging.info(f"Validation loss: {val_loss:.4f}")
        logging.info(f"Validation metrics: {val_metrics}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            model.save_model(config['output_dir'] / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                logging.info("Early stopping triggered")
                break
    
    # Final evaluation on test set
    model = PhishingDetector.load_model(config['output_dir'] / 'best_model.pt', config['model_name'])
    test_loss, test_metrics = evaluate(model, test_loader, test_dataset, device, criterion)
    logging.info(f"Test metrics: {test_metrics}")
    
    # Update README with metrics
    update_readme_metrics(test_metrics)

def update_readme_metrics(metrics: Dict[str, float]) -> None:
    """Update README.md with model performance metrics."""
    metrics_text = f"""
## Model Performance

- Accuracy: {metrics['accuracy']:.4f}
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1 Score: {metrics['f1']:.4f}
"""
    
    try:
        with open("README.md", "r") as f:
            content = f.read()
        
        # Replace the metrics section or append if not found
        if "## Model Performance" in content:
            parts = content.split("## Model Performance")
            content = parts[0] + metrics_text
        else:
            content += metrics_text
        
        with open("README.md", "w") as f:
            f.write(content)
            
        logging.info("Updated README.md with model metrics")
    except Exception as e:
        logging.error(f"Error updating README: {str(e)}")

if __name__ == "__main__":
    config = {
        'model_name': 'prajjwal1/bert-tiny',
        'data_path': 'data/phishing_dataset.csv',
        'output_dir': Path('models'),
        'batch_size': 32,
        'learning_rate': 2e-5,
        'epochs': 2,
        'patience': 2
    }
    
    # Create output directory if it doesn't exist
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    
    train(config) 
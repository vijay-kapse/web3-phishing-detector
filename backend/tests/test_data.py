import pytest
import pandas as pd
from transformers import AutoTokenizer
import os

from src.data.dataset import PhishingDataset
from src.data.preprocess import preprocess_text, is_web3_related

@pytest.fixture
def sample_data(tmp_path):
    """Create a sample dataset for testing."""
    data = pd.DataFrame({
        'text': [
            'Message about ethereum wallet',
            'Regular spam message',
            'Crypto mining opportunity',
            'Normal business email'
        ],
        'label': [1, 1, 1, 0]
    })
    
    path = tmp_path / "test_data.csv"
    data.to_csv(path, index=False)
    return str(path)

@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")

def test_dataset_loading(sample_data, tokenizer):
    """Test dataset loading functionality."""
    dataset = PhishingDataset(sample_data, tokenizer)
    data = dataset.load_data()
    
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 4
    assert all(col in data.columns for col in ['text', 'label'])

def test_data_splitting(sample_data, tokenizer):
    """Test data splitting functionality."""
    dataset = PhishingDataset(sample_data, tokenizer)
    train, val, test = dataset.split_data(test_size=0.25, val_size=0.25)
    
    # Check sizes
    assert len(train) + len(val) + len(test) == 4
    assert all(isinstance(df, pd.DataFrame) for df in [train, val, test])

def test_text_preprocessing():
    """Test text preprocessing function."""
    text = "  UPPER case Message  "
    processed = preprocess_text(text)
    
    assert processed == "upper case message"
    assert preprocess_text(None) == ""
    assert preprocess_text(123) == "123"

def test_web3_detection():
    """Test Web3-related content detection."""
    assert is_web3_related("Message about ethereum")
    assert is_web3_related("NFT marketplace launch")
    assert is_web3_related("Your crypto wallet needs attention")
    assert not is_web3_related("Regular spam message")
    assert not is_web3_related("Business proposal")

def test_tokenizer_preprocessing(tokenizer):
    """Test tokenizer preprocessing."""
    dataset = PhishingDataset("dummy_path", tokenizer)
    inputs = dataset.preprocess("Test message")
    
    assert "input_ids" in inputs
    assert "attention_mask" in inputs
    assert inputs["input_ids"].shape[1] <= 512  # Max length
    assert inputs["attention_mask"].shape == inputs["input_ids"].shape 
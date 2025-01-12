import pytest
import torch
from transformers import AutoTokenizer
import numpy as np

from src.models.phishing_detector import PhishingDetector

@pytest.fixture
def model():
    return PhishingDetector()

@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")

def test_model_output_shape(model, tokenizer):
    """Test if model outputs correct shape."""
    text = "Test message about crypto wallet"
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        output = model(**inputs)
    
    assert output.shape == (1, 1)
    assert 0 <= output.item() <= 1

def test_model_batch_processing(model, tokenizer):
    """Test if model can handle batched inputs."""
    texts = [
        "First message about ethereum",
        "Second message about bitcoin"
    ]
    inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    assert outputs.shape == (2, 1)
    assert all(0 <= p <= 1 for p in outputs.squeeze().tolist())

def test_model_save_load(model, tmp_path):
    """Test model save and load functionality."""
    # Save model
    save_path = tmp_path / "test_model.pt"
    model.save_model(save_path)
    
    # Load model
    loaded_model = PhishingDetector.load_model(save_path)
    
    # Compare parameters
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.equal(p1, p2)

def test_model_predict_method(model, tokenizer):
    """Test the predict convenience method."""
    text = "Message about NFT marketplace"
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    prediction = model.predict(inputs)
    
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (1,)
    assert 0 <= prediction[0] <= 1 
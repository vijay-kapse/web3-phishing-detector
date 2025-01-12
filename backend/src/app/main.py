import logging
import sys
from pathlib import Path
from flask import Flask, request, jsonify, current_app
from transformers import AutoTokenizer
import torch
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path to fix imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.phishing_detector import PhishingDetector

def create_app():
    app = Flask(__name__)
    
    # Load model and tokenizer
    MODEL_PATH = str(project_root / "models/best_model.pt")
    MODEL_NAME = "prajjwal1/bert-tiny"
    
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Model path: {MODEL_PATH}")
    
    # Initialize model and tokenizer at startup
    try:
        logger.info("Loading tokenizer and model...")
        app.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        logger.info("Tokenizer loaded successfully")
        
        app.model = PhishingDetector.load_model(MODEL_PATH, MODEL_NAME)
        app.model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
    
    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            logger.debug("Received prediction request")
            data = request.get_json()
            
            if not data or 'text' not in data:
                logger.error("No text provided in request")
                return jsonify({"error": "No text provided"}), 400
                
            text = data["text"]
            logger.debug(f"Input text: {text[:100]}...")
            
            # Preprocess
            logger.debug("Tokenizing input")
            inputs = current_app.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Inference
            logger.debug("Running inference")
            with torch.no_grad():
                prediction = current_app.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                probability = prediction.item()
                is_phishing = probability >= 0.5
                
            logger.info(f"Prediction: is_phishing={is_phishing}, probability={probability:.4f}")
            
            return jsonify({
                "is_phishing": bool(is_phishing),
                "probability": probability
            })
        
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return jsonify({"error": str(e)}), 400

    @app.route("/")
    def home():
        try:
            logger.debug("Serving index.html")
            return app.send_static_file("index.html")
        except Exception as e:
            logger.error(f"Error serving index.html: {str(e)}")
            return jsonify({"error": "Failed to load interface"}), 500
            
    @app.route("/health")
    def health():
        return jsonify({"status": "healthy", "model_loaded": hasattr(current_app, "model")})
    
    return app

if __name__ == "__main__":
    logger.info("Starting Flask application...")
    app = create_app()
    app.run(debug=True, port=5000) 
from http.server import BaseHTTPRequestHandler
import json
import torch
from transformers import AutoTokenizer
from ..src.models.phishing_detector import PhishingDetector

MODEL_NAME = "prajjwal1/bert-tiny"
MODEL_PATH = "models/best_model.pt"

# Initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = PhishingDetector.load_model(MODEL_PATH)
model.eval()

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data)
        
        try:
            text = data.get('text')
            if not text:
                self._send_response(400, {'error': 'No text provided'})
                return
                
            # Tokenize and predict
            inputs = tokenizer(
                text,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.sigmoid(outputs).numpy()
                
            prediction = bool(probabilities[0] > 0.5)
            confidence = float(probabilities[0] if prediction else 1 - probabilities[0])
            
            response = {
                'is_phishing': prediction,
                'confidence': confidence,
                'text': text
            }
            
            self._send_response(200, response)
            
        except Exception as e:
            self._send_response(500, {'error': str(e)})
    
    def _send_response(self, status_code, data):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode()) 
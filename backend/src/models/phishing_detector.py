import torch
import torch.nn as nn
from transformers import AutoModel

class PhishingDetector(nn.Module):
    def __init__(self, model_name='prajjwal1/bert-tiny'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[0][:, 0]  # Take CLS token output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return torch.sigmoid(logits)

    def predict(self, text, tokenizer):
        encoding = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            outputs = self(input_ids, attention_mask)
            
        return outputs.numpy()

    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    @classmethod
    def load_model(cls, path, model_name='prajjwal1/bert-tiny'):
        model = cls(model_name)
        model.load_state_dict(torch.load(path))
        return model 
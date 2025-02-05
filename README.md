# Web3 Phishing Detection System
<img width="1001" alt="Screenshot 2025-01-12 at 11 38 24 PM" src="https://github.com/user-attachments/assets/8a738c8f-5238-4994-8283-354b1b9520b3" />


An end-to-end machine learning system for detecting phishing messages targeting Web3 users.

## Project Structure
```
├── src/
│   ├── data/          # Data processing and loading
│   ├── models/        # Model architecture and components
│   ├── training/      # Training pipeline
│   ├── utils/         # Helper functions
│   └── app/           # Flask application
├── data/              # Dataset storage
└── tests/             # Unit tests
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Pipeline
To train the model:
```bash
python src/training/train.py
```

### Running the Application
Start the Flask server:
```bash
python src/app/main.py
```

Visit http://localhost:5000 to access the UI.

## Development

- Format code: `black src/`
- Run tests: `pytest tests/`
- Lint code: `flake8 src/`


## Model Performance

- Accuracy: 0.6667
- Precision: 0.7826
- Recall: 0.4737
- F1 Score: 0.5902

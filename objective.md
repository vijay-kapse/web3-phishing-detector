Objective
Develop an end-to-end training pipeline to detect phishing messages targeting Web3 users. The solution must train a Large Language Model (LLM) on a phishing dataset, package the trained model into a Flask application, and provide a simple UI for inference.

Scope & Requirements
Data & Model Training

Use a Web3 phishing dataset to train the model.
Implement training in Python scripts (no Jupyter notebooks).
Choose an appropriate machine learning or deep learning algorithm.
Ensure code follows clean coding practices for reusability and reproducibility.
Model Packaging & Inference

Package the trained model in a Flask application.
Provide both binary and probability score outputs.
User Interface

Implement a minimal UI (input box + run button).
Show prediction result (phishing or not) or probability score in real-time.
Deliverables

Python scripts for:
End-to-end training pipeline
Loading the trained model and performing inference
Flask app for serving the model
UI (HTML + JavaScript or any simple frontend framework)
Evaluation metrics documented in a README (accuracy, precision, recall, F1-score, or other relevant metrics)
README with instructions for:
Training pipeline execution
Running the Flask application
requirements.txt for all dependencies
Acceptance Criteria
A fully functional training pipeline that outputs a trained model.
Flask application runs locally, accepts an input message, and returns a prediction or probability.
Clear, concise documentation for setup and usage.
Code follows clean coding standards and best practices.
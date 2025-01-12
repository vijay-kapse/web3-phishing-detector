from setuptools import setup, find_packages

setup(
    name="web3-phishing-detector",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "scikit-learn>=1.0.2",
        "pandas>=1.5.0",
        "numpy==1.24.3",
        "flask>=2.0.0",
        "python-dotenv>=0.19.0",
        "wandb>=0.15.0"
    ],
) 
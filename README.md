# Transparent Deep Learning for Credit Risk Prediction

## Project Overview
This repository contains a PyTorch-based Feature Tokenizer-Transformer (FT-Transformer) designed for high-stakes credit risk prediction. It compares deep learning architecture against traditional machine learning baselines (Logistic Regression, Random Forest) on tabular financial data.

## 1. Environment Setup

Run this project within a Python virtual environment.

**Prerequisites:**
* Python 3.12 

**Installation:**
Clone the repository and navigate to the project root:
   ```bash
   cd GroupProject-
  ```
Create and activate a virtual environment:

Bash
python -m venv venv then .\venv\Scripts\activate

Install the required dependencies:

  ```bash
  pip install -r requirements.txt
  ```


Data & Pre-trained Models

Serialized Assets
To run Demo.py without retraining, the following serialized files must be present in your directory (these are automatically generated when you successfully run ft_transformer.py once):

ft_transformer_weights.pth: The optimized neural network weights.

model_config.pt: The dynamic architectural blueprint (e.g., feature counts).

sample_applicant_data.pt: A preprocessed, scaled tensor of a single unseen applicant.

This script bypasses the training loop, loads the pre-trained weights, and evaluates an unseen applicant instantly.

  ```Bash
  python Demo.py
  ```


Model Training & Evaluation:
If you want to run the full data preprocessing pipeline, train the FT-Transformer yourself and produce SHAP plots

  ```Bash
  cd src
  ```

# Run the main training script
  ```bash
  python -m models/ft_transformer_revised
  ```
# Produce SHAP plots
  ```bash
  python explainability/shap_ft_transformer
  ```


# Lightweight and Explainable Phishing URL Detection Using Machine Learning

## Project
This project detects whether a URL is **phishing** or **legitimate** using URL-based lexical features.

It includes:
- URL feature extraction
- Data preprocessing
- Logistic Regression, Decision Tree and Random Forest
- Evaluation with accuracy, precision, recall and F1-score
- Confusion matrix and feature-importance plot
- Explainable prediction for an individual URL
- Simple Streamlit web interface

## 1. Install
```bash
pip install -r requirements.txt
```

## 2. Dataset
Put your dataset at:
`dataset/urls.csv`

Required columns:
- `url`
- `label`

Labels can be:
- `phishing` / `legitimate`
- `1` / `0`

Example:
```csv
url,label
https://www.google.com,legitimate
http://example-login.com/verify,phishing
```

A tiny sample is included only for testing the pipeline. Replace it with a real dataset for your final results.

## 3. Train
```bash
python src/train_model.py
```

This creates:
- `models/phishing_model.pkl`
- `models/feature_names.json`
- `results/model_comparison.csv`
- `results/confusion_matrix.png`
- `results/feature_importance.png`

## 4. Test a URL
```bash
python src/predict.py "https://www.google.com"
```

## 5. Run the interface
```bash
streamlit run app.py
```

## Academic note
Do not report the sample-dataset accuracy in your final report. Train on a sufficiently large, representative phishing/legitimate dataset and report the actual test results.

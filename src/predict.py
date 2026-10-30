import os
import sys
import poplib

import joblib

try:
    import pandas as pd  # type: ignore[import-not-found]
except ModuleNotFoundError:
    pd = None  # type: ignore[assignment]

from feature_extraction import extract_features

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "models", "phishing_model.pkl")

def explain(features):
    reasons = []
    if features["has_ip"]:
        reasons.append("URL contains an IP address")
    if not features["has_https"]:
        reasons.append("URL does not use HTTPS")
    if features["suspicious_keyword_count"] > 0:
        reasons.append("Suspicious keyword(s) detected")
    if features["url_length"] > 75:
        reasons.append("URL is unusually long")
    if features["num_hyphens"] >= 3:
        reasons.append("Many hyphens are present")
    if features["num_at"] > 0:
        reasons.append("The @ symbol is present")
    if features["num_special_chars"] >= 5:
        reasons.append("Many special characters are present")
    if features["subdomain_count"] >= 2:
        reasons.append("Multiple subdomains are present")
    if not reasons:
        reasons.append("No strong lexical warning signs were detected")
    return reasons

def predict_url(url):
    model = joblib.load(MODEL_PATH)
    features = extract_features(url)
    X = pd.DataFrame([features])

    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None

    label = "PHISHING" if pred == 1 else "LEGITIMATE"

    print(f"\nURL: {url}")
    print(f"Prediction: {label}")
    if proba is not None:
        print(f"Phishing probability: {proba:.2%}")

    print("\nExplanation:")
    for reason in explain(features):
        print(f"- {reason}")

    return label, features

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python src/predict.py "https://example.com"')
        sys.exit(1)
    predict_url(sys.argv[1])

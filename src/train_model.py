import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from feature_extraction import extract_dataframe

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "dataset", "PhiUSIIL_Phishing_URL_Dataset.csv")
MODEL_DIR = os.path.join(BASE, "models")
RESULT_DIR = os.path.join(BASE, "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

df = pd.read_csv(DATA)
df.columns = df.columns.str.strip()
df = df.rename(columns={"URL": "url"})
df = df.dropna(subset=["url", "label"]).drop_duplicates(subset=["url"])

def label_to_int(x):
    s = str(x).strip().lower()

    if s in {"0", "phishing", "malicious", "bad"}:
        return 1

    if s in {"1", "legitimate", "benign", "good"}:
        return 0

    raise ValueError(f"Unknown label: {x}")

df["target"] = df["label"].map(label_to_int)

X = extract_dataframe(df["url"])
y = df["target"]

if y.nunique() < 2:
    raise ValueError("Dataset must contain both phishing and legitimate URLs.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    )
}

rows = []
trained = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0)
    })
    trained[name] = model

results = pd.DataFrame(rows).sort_values("F1", ascending=False)
results.to_csv(os.path.join(RESULT_DIR, "model_comparison.csv"), index=False)

best_name = results.iloc[0]["Model"]
best_model = trained[best_name]
joblib.dump(best_model, os.path.join(MODEL_DIR, "phishing_model.pkl"))

with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as f:
    json.dump(list(X.columns), f, indent=2)

pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legitimate", "Phishing"],
            yticklabels=["Legitimate", "Phishing"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"), dpi=200)
plt.close()

# Feature importance where supported
model_obj = best_model.named_steps["model"] if hasattr(best_model, "named_steps") else best_model
importance = None
if hasattr(model_obj, "feature_importances_"):
    importance = model_obj.feature_importances_
elif hasattr(model_obj, "coef_"):
    importance = abs(model_obj.coef_[0])

if importance is not None:
    fi = pd.DataFrame({"Feature": X.columns, "Importance": importance})
    fi = fi.sort_values("Importance", ascending=False).head(15)
    fi.to_csv(os.path.join(RESULT_DIR, "feature_importance.csv"), index=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=fi, x="Importance", y="Feature")
    plt.title(f"Top Features - {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "feature_importance.png"), dpi=200)
    plt.close()

print("\nModel comparison:")
print(results.to_string(index=False))
print(f"\nSelected model: {best_name}")
print("Saved trained model to models/phishing_model.pkl")

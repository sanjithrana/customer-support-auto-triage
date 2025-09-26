# src/evaluate.py
import joblib
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from preprocess import load_and_prepare

def evaluate(model_path: str, data_path: str, label_col: str = "Category"):
    print("Loading data...")
    df = load_and_prepare(data_path)
    X = df["text"].values
    y = df[label_col].values
    print(f"Loaded {len(df)} samples.")
    model = joblib.load(model_path)
    # measure latency per sample (average)
    n = min(1000, len(X))
    sample_X = X[:n]
    start = time.time()
    preds = model.predict(sample_X)
    duration = time.time() - start
    avg_latency_ms = (duration / n) * 1000.0
    print(f"Average inference latency (ms/sample) on {n} samples: {avg_latency_ms:.3f} ms")
    # full predictions
    preds_full = model.predict(X)
    acc = accuracy_score(y, preds_full)
    precision, recall, f1, _ = precision_recall_fscore_support(y, preds_full, average='macro', zero_division=0)
    print("Accuracy:", acc)
    print("Precision (macro):", precision)
    print("Recall (macro):", recall)
    print("F1 (macro):", f1)
    print("Classification Report:\n", classification_report(y, preds_full, zero_division=0))
    cm = confusion_matrix(y, preds_full, labels=np.unique(y))
    print("Confusion matrix shape:", cm.shape)
    # Save detailed report
    report = {
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "avg_latency_ms": float(avg_latency_ms)
    }
    joblib.dump(report, model_path.replace(".joblib", "_eval_report.joblib"))
    print("Saved evaluation report next to model.")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/ticket_model.joblib")
    parser.add_argument("--data_path", type=str, default="data/tickets.csv")
    parser.add_argument("--label_col", type=str, default="Category")
    args = parser.parse_args()
    evaluate(args.model_path, args.data_path, args.label_col)

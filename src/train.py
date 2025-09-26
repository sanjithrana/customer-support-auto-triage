# src/train.py
import os
import argparse
import joblib
import time
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix

from preprocess import load_and_prepare

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train(args):
    df = load_and_prepare(args.data_path)
    X = df["text"].values
    y = df[args.label_col].values
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="saga"))
    ])
    # grid search (small)
    param_grid = {
        "tfidf__max_features": [5000, 10000, 20000],
        "clf__C": [0.1, 1.0, 5.0]
    }
    gs = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=2, scoring="f1_macro")
    print("Starting grid search...")
    start = time.time()
    gs.fit(X_train, y_train)
    duration = time.time() - start
    print(f"Grid search done in {duration:.1f}s. Best params: {gs.best_params_}")
    best_model = gs.best_estimator_
    # Evaluate on test
    preds = best_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    report = classification_report(y_test, preds)
    print("Test Accuracy:", acc)
    print("Test F1 (macro):", f1)
    print("Classification Report:\n", report)
    # Save model
    model_path = os.path.join(MODEL_DIR, f"ticket_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"Saved model to {model_path}")
    # Save a quick evaluation CSV
    eval_df = pd.DataFrame({"text": X_test, "true": y_test, "pred": preds})
    eval_df.to_csv(os.path.join(MODEL_DIR, "eval_results.csv"), index=False)
    # Save metadata
    metadata = {
        "best_params": gs.best_params_,
        "accuracy_test": float(acc),
        "f1_macro_test": float(f1),
        "training_time_sec": duration
    }
    joblib.dump(metadata, os.path.join(MODEL_DIR, "metadata.joblib"))
    print("Training complete.")
    return model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/tickets.csv")
    parser.add_argument("--label_col", type=str, default="Category")
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()
    train(args)

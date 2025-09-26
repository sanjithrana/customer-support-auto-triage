# src/preprocess.py
import re
import pandas as pd
import numpy as np
from typing import Tuple, List
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# download NLTK data on first run
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """Basic text cleaning pipeline."""
    if pd.isna(text):
        return ""
    # lower
    text = text.lower()
    # replace URLs, emails, ticket ids etc.
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    # remove non-alphanumeric (keep spaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # tokenize and remove stopwords, lemmatize
    tokens = [LEMMATIZER.lemmatize(tok) for tok in text.split() if tok not in STOPWORDS]
    return " ".join(tokens)

def load_and_prepare(path: str, text_cols: List[str] = ["Subject", "Description"], label_col: str = "Category") -> pd.DataFrame:
    """
    Loads CSV and produces a DataFrame with a 'text' column (combined & cleaned)
    and label column.
    """
    df = pd.read_csv(path)
    # Basic sanity checks
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset columns: {df.columns.tolist()}")
    # Fill NaNs
    for col in text_cols:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("")
    # Create combined text
    df["text"] = (df["Subject"].astype(str) + " " + df["Description"].astype(str)).apply(clean_text)
    # Optional: encode priority as categorical ordinal (low, medium, high)
    if "Priority" in df.columns:
        df["Priority_enc"] = df["Priority"].fillna("unknown").astype(str)
        # simple mapping; you can expand
        mapping = {"low": 0, "medium": 1, "high": 2}
        df["Priority_enc"] = df["Priority_enc"].str.lower().map(mapping).fillna(0).astype(int)
    else:
        df["Priority_enc"] = 0
    # Drop rows with empty text or missing label
    df = df[df["text"].str.strip() != ""]
    df = df[df[label_col].notna()]
    df = df.reset_index(drop=True)
    return df

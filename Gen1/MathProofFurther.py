import os
import re
import time
import random
import numpy as np
import pandas as pd
import spacy
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from scipy.stats import norm

# ---------------------------
# Load full spaCy model (with dependency parsing and NER)
# ---------------------------
nlp = spacy.load("en_core_web_sm")  # full model with dependency parsing and NER enabled


# ---------------------------
# Data Loader for protests.csv
# ---------------------------
def load_protests_data(csv_path):
    """
    Expects a CSV file with:
      - A text column (one of: "comment_text", "text", "body", "content", "Event (legacy; see tags)")
      - Optionally a "label" column. If missing and "Tags" exists, binary labels are created.
    """
    df = pd.read_csv(csv_path)
    candidate_text_columns = ["comment_text", "text", "body", "content", "Event (legacy; see tags)"]
    text_col = None
    for col in candidate_text_columns:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        available = list(df.columns)
        raise Exception(
            f"No suitable text column found. Expected one of {candidate_text_columns}. Available columns: {available}")
    if 'label' not in df.columns:
        if 'Tags' in df.columns:
            df['label'] = df['Tags'].apply(lambda x: 1 if "protest" in str(x).lower() else 0)
            print("No 'label' column found. Created binary labels from 'Tags'.")
        else:
            raise Exception("Expected a column named 'label' or 'Tags' to derive labels.")
    df = df.dropna(subset=[text_col, 'label'])
    df['label'] = df['label'].astype(int)
    df = df.rename(columns={text_col: 'comment_text'})
    return df


# ---------------------------
# Balance the Dataset (undersample majority)
# ---------------------------
def balance_dataset(df):
    counts = df['label'].value_counts()
    min_count = counts.min()
    balanced_list = []
    for label in counts.index:
        balanced_list.append(df[df['label'] == label].sample(n=min_count, random_state=42))
    return pd.concat(balanced_list).sample(frac=1, random_state=42)  # shuffle


# ===========================
# Emergent Concept Analyzer
# ===========================
def emergent_concept_analyzer(text):
    """
    Processes text with spaCy and returns a list of emergent concept strings.
    Extracts noun chunks and named entities.
    """
    doc = nlp(text)
    noun_chunks = [chunk.text.strip() for chunk in doc.noun_chunks]
    entities = [ent.text.strip() for ent in doc.ents]
    concepts = [c for c in noun_chunks + entities if c]  # filter out empty strings
    return concepts


# ===========================
# Pipeline Comparison Function
# ===========================
def run_comparison(data, test_size=0.3, random_state=42):
    # Balance dataset
    balanced_data = balance_dataset(data)

    # Split into train and test sets
    X = balanced_data['comment_text']
    y = balanced_data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state, stratify=y)

    # ------------------------------
    # Emergent Concepts Pipeline
    # ------------------------------
    emergent_vectorizer = CountVectorizer(analyzer=emergent_concept_analyzer, lowercase=True)
    X_train_emergent = emergent_vectorizer.fit_transform(X_train)
    X_test_emergent = emergent_vectorizer.transform(X_test)

    clf_emergent = LogisticRegression(solver='liblinear', random_state=random_state)
    clf_emergent.fit(X_train_emergent, y_train)

    y_pred_emergent = clf_emergent.predict(X_test_emergent)
    y_proba_emergent = clf_emergent.predict_proba(X_test_emergent)[:, 1]

    emergent_report = classification_report(y_test, y_pred_emergent, zero_division=0)
    emergent_cm = confusion_matrix(y_test, y_pred_emergent)
    emergent_roc_auc = roc_auc_score(y_test, y_proba_emergent)
    emergent_acc = accuracy_score(y_test, y_pred_emergent)
    emergent_f1 = f1_score(y_test, y_pred_emergent, average="macro", zero_division=0)

    # ------------------------------
    # Baseline Pipeline (Default Tokenization)
    # ------------------------------
    baseline_vectorizer = CountVectorizer(
        lowercase=True)  # default tokenization, stopword removal can be added if desired
    X_train_baseline = baseline_vectorizer.fit_transform(X_train)
    X_test_baseline = baseline_vectorizer.transform(X_test)

    clf_baseline = LogisticRegression(solver='liblinear', random_state=random_state)
    clf_baseline.fit(X_train_baseline, y_train)

    y_pred_baseline = clf_baseline.predict(X_test_baseline)
    y_proba_baseline = clf_baseline.predict_proba(X_test_baseline)[:, 1]

    baseline_report = classification_report(y_test, y_pred_baseline, zero_division=0)
    baseline_cm = confusion_matrix(y_test, y_pred_baseline)
    baseline_roc_auc = roc_auc_score(y_test, y_proba_baseline)
    baseline_acc = accuracy_score(y_test, y_pred_baseline)
    baseline_f1 = f1_score(y_test, y_pred_baseline, average="macro", zero_division=0)

    # ------------------------------
    # Print and Compare Results
    # ------------------------------
    print("\n--- Emergent Concepts Pipeline ---")
    print(emergent_report)
    print("Confusion Matrix:")
    print(emergent_cm)
    print(f"Accuracy: {emergent_acc:.4f}")
    print(f"Macro F1: {emergent_f1:.4f}")
    print(f"ROC-AUC: {emergent_roc_auc:.4f}")

    print("\n--- Baseline Pipeline ---")
    print(baseline_report)
    print("Confusion Matrix:")
    print(baseline_cm)
    print(f"Accuracy: {baseline_acc:.4f}")
    print(f"Macro F1: {baseline_f1:.4f}")
    print(f"ROC-AUC: {baseline_roc_auc:.4f}")

    # Plot ROC curves for comparison
    from sklearn.metrics import roc_curve
    fpr_e, tpr_e, _ = roc_curve(y_test, y_proba_emergent)
    fpr_b, tpr_b, _ = roc_curve(y_test, y_proba_baseline)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_e, tpr_e, label=f'Emergent ROC (AUC = {emergent_roc_auc:.4f})', lw=2)
    plt.plot(fpr_b, tpr_b, label=f'Baseline ROC (AUC = {baseline_roc_auc:.4f})', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    return {
        "emergent": {
            "classifier": clf_emergent,
            "vectorizer": emergent_vectorizer,
            "report": emergent_report,
            "confusion_matrix": emergent_cm,
            "accuracy": emergent_acc,
            "macro_f1": emergent_f1,
            "roc_auc": emergent_roc_auc
        },
        "baseline": {
            "classifier": clf_baseline,
            "vectorizer": baseline_vectorizer,
            "report": baseline_report,
            "confusion_matrix": baseline_cm,
            "accuracy": baseline_acc,
            "macro_f1": baseline_f1,
            "roc_auc": baseline_roc_auc
        }
    }


# ===========================
# Main Script
# ===========================
def main():
    DATASET_FILENAME = "protests.csv"  # Ensure this CSV is in your working directory.
    data = load_protests_data(DATASET_FILENAME)
    print("Full Dataset summary (label counts):")
    print(data['label'].value_counts())

    print("\nComparing emergent concept detection against baseline token-based approach...")
    start_time = time.time()
    results = run_comparison(data)
    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds")


if __name__ == '__main__':
    main()

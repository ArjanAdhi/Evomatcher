import os
import re
import time
import numpy as np
import pandas as pd
import spacy
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ---------------------------
# Load full spaCy model
# ---------------------------
nlp = spacy.load("en_core_web_sm")


# ---------------------------
# Load Dataset
# ---------------------------
def load_protests_data(csv_path):
    df = pd.read_csv(csv_path)
    candidate_text_columns = ["comment_text", "text", "body", "content", "Event (legacy; see tags)"]
    text_col = next((col for col in candidate_text_columns if col in df.columns), None)
    if not text_col:
        raise Exception(f"No suitable text column found. Available: {list(df.columns)}")
    if 'label' not in df.columns and 'Tags' in df.columns:
        df['label'] = df['Tags'].apply(lambda x: 1 if "protest" in str(x).lower() else 0)
        print("No 'label' column found. Created binary labels from 'Tags'.")
    df = df.dropna(subset=[text_col, 'label'])
    df['label'] = df['label'].astype(int)
    df = df.rename(columns={text_col: 'comment_text'})
    return df


# ---------------------------
# Balance the Dataset
# ---------------------------
def balance_dataset(df):
    min_count = df['label'].value_counts().min()
    return pd.concat(
        [df[df['label'] == label].sample(n=min_count, random_state=42) for label in df['label'].unique()]).sample(
        frac=1, random_state=42)


# ===========================
# Dual-Branch Architecture Functions
# ===========================

def concept_mapping_score(text):
    """ Extracts hierarchical concept mappings using spaCy. """
    doc = nlp(text)
    return [chunk.text.strip() for chunk in doc.noun_chunks] + [ent.text.strip() for ent in doc.ents]


def statistical_connection_score(text):
    """ Extracts statistical co-occurrence signals using keyword frequencies. """
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]


def dual_branch_analyzer(text):
    """ Combines concept mapping + statistical analysis for feature extraction. """
    return concept_mapping_score(text) + statistical_connection_score(text)


# ===========================
# Model Training & Comparison
# ===========================

def train_and_evaluate(X_train, X_test, y_train, y_test, vectorizer, model_name):
    """ Trains a logistic regression classifier and evaluates its performance. """
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(solver='liblinear', random_state=42)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    y_proba = clf.predict_proba(X_test_vec)[:, 1]

    print(f"\n--- {model_name} ---")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_proba)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    return clf, vectorizer, y_proba, roc_auc


# ===========================
# Main Script: Run All Models
# ===========================

def main():
    DATASET_FILENAME = "protests.csv"
    data = load_protests_data(DATASET_FILENAME)
    print("\nFull Dataset summary:")
    print(data['label'].value_counts())

    balanced_data = balance_dataset(data)
    print("\nBalanced Dataset summary:")
    print(balanced_data['label'].value_counts())

    # Split into train/test
    X = balanced_data['comment_text']
    y = balanced_data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Define all models
    models = {
        "Baseline (Token-Based)": CountVectorizer(lowercase=True),
        "Emergent Concept-Based": CountVectorizer(analyzer=concept_mapping_score, lowercase=True),
        "Dual-Branch Evolutionary": CountVectorizer(analyzer=dual_branch_analyzer, lowercase=True)
    }

    results = {}

    # Train and evaluate all models
    for model_name, vectorizer in models.items():
        clf, vec, y_proba, roc_auc = train_and_evaluate(X_train, X_test, y_train, y_test, vectorizer, model_name)
        results[model_name] = (clf, vec, y_proba, roc_auc)

    # Plot ROC curves for all models
    plt.figure(figsize=(8, 6))
    for model_name, (_, _, y_proba, roc_auc) in results.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})', lw=2)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()

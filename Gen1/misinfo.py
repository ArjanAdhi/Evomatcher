import os
import numpy as np
import pandas as pd
import spacy
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, f1_score, accuracy_score)
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# ---------------------------
# Load spaCy globally in main process
# ---------------------------
nlp = spacy.load("en_core_web_sm")


# ---------------------------
# Worker initializer: each worker loads its own spaCy model
# ---------------------------
def init_worker():
    global nlp
    nlp = spacy.load("en_core_web_sm")


# ---------------------------
# Load Politifact Fact Check Dataset from JSON and Process It
# ---------------------------
def load_factcheck_data(json_path):
    """
    Loads the Politifact Fact Check Dataset from a JSON file.
    Expects records with at least the following fields:
      - "verdict": one of {"true", "mostly-true", "half-true", "mostly-false", "false", "pants-fire"}
      - "statement": the statement that was fact-checked.
    Converts the multi-class verdict into a binary label:
      - 0 (true/real) if verdict in {"true", "mostly-true", "half-true"}
      - 1 (fake) if verdict in {"mostly-false", "false", "pants-fire"}
    Renames "statement" to "comment_text" for consistency.
    """
    # Use lines=True for newline-delimited JSON
    df = pd.read_json(json_path, orient="records", lines=True)
    required_cols = ["verdict", "statement"]
    for col in required_cols:
        if col not in df.columns:
            raise Exception(f"Dataset must contain '{col}' column.")

    # Create binary label using a mapping
    true_verdicts = {"true", "mostly-true", "half-true"}
    fake_verdicts = {"mostly-false", "false", "pants-fire"}

    def map_verdict(verdict):
        verdict = verdict.lower().strip()
        if verdict in true_verdicts:
            return 0
        elif verdict in fake_verdicts:
            return 1
        else:
            # For any unexpected value, return np.nan so it can be dropped
            return np.nan

    df['label'] = df['verdict'].apply(map_verdict)
    # Drop rows with missing labels or statements
    df = df.dropna(subset=["statement", "label"])
    df['label'] = df['label'].astype(int)
    # Rename 'statement' to 'comment_text'
    df = df.rename(columns={'statement': 'comment_text'})
    return df


# ---------------------------
# (Optional) Balance the Dataset
# ---------------------------
def balance_dataset(df):
    """
    Balances the dataset so that each class has an equal number of samples.
    """
    min_count = df['label'].value_counts().min()
    balanced_list = [df[df['label'] == label].sample(n=min_count, random_state=42)
                     for label in df['label'].unique()]
    return pd.concat(balanced_list).sample(frac=1, random_state=42)


# ---------------------------
# Dual-Branch Feature Extraction Functions
# ---------------------------
def concept_mapping_score(text):
    """Extract noun chunks and named entities."""
    doc = nlp(text)
    return [chunk.text.strip() for chunk in doc.noun_chunks] + [ent.text.strip() for ent in doc.ents]


def statistical_connection_score(text):
    """Extract lemmatized tokens excluding stop words and punctuation."""
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]


def dual_branch_analyzer(text):
    """Combine concept mapping and statistical connection extraction."""
    return concept_mapping_score(text) + statistical_connection_score(text)


# ---------------------------
# Parallel Feature Extraction using Multiprocessing
# ---------------------------
def parallel_feature_extraction(texts, analyzer_function):
    """
    Uses all available CPU cores to process texts in parallel.
    """
    num_workers = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
        results = list(executor.map(analyzer_function, texts))
    return results


# ---------------------------
# Explainability Function: Show Token Contributions
# ---------------------------
def explain_prediction(text, vectorizer, clf):
    """
    Analyzes an input text and displays which tokens contributed most to its classification.
    """
    tokens = dual_branch_analyzer(text)
    token_string = " ".join(tokens)
    vector = vectorizer.transform([token_string])

    try:
        feature_names = vectorizer.get_feature_names_out()
    except AttributeError:
        feature_names = vectorizer.get_feature_names()

    importances = clf.feature_importances_
    token_contrib = {}
    for token in tokens:
        if token in vectorizer.vocabulary_:
            idx = vectorizer.vocabulary_[token]
            count = vector[0, idx]
            token_contrib[token] = count * importances[idx]

    if token_contrib:
        sorted_tokens = sorted(token_contrib.items(), key=lambda x: x[1], reverse=True)
        print("\nTop Token Contributions to Classification:")
        for token, contrib in sorted_tokens[:10]:  # Show top 10 tokens
            print(f"{token}: {contrib:.4f}")

        # Plot top 10 token contributions
        tokens_plot, contribs_plot = zip(*sorted_tokens[:10])
        plt.figure(figsize=(8, 4))
        plt.bar(tokens_plot, contribs_plot)
        plt.title("Token Contribution Breakdown")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
    else:
        print("No tokens from the input text were found in the model vocabulary.")


# ---------------------------
# Main Application: Real-World Test on Politifact Dataset
# ---------------------------
def main():
    JSON_FILENAME = "fact.json"  # Path to your Politifact JSON file
    data = load_factcheck_data(JSON_FILENAME)

    print("\nDataset Label Distribution:")
    print(data['label'].value_counts())

    # (Optional) Balance the dataset if desired.
    # data = balance_dataset(data)

    # --- Train/Test Split for Real Accuracy ---
    print("\nSplitting data into train/test sets...")
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])
    print(f"Train set: {train_df.shape}  Test set: {test_df.shape}")

    # --- Parallel Feature Extraction for Training Data ---
    print("\nExtracting features (train) using all CPU cores...")
    train_tokens = parallel_feature_extraction(train_df['comment_text'], dual_branch_analyzer)
    X_train_list = [" ".join(tokens) for tokens in train_tokens]

    # --- Parallel Feature Extraction for Testing Data ---
    print("Extracting features (test) using all CPU cores...")
    test_tokens = parallel_feature_extraction(test_df['comment_text'], dual_branch_analyzer)
    X_test_list = [" ".join(tokens) for tokens in test_tokens]

    # --- Vectorization ---
    vectorizer = CountVectorizer(lowercase=True)
    X_train_vec = vectorizer.fit_transform(X_train_list)
    X_test_vec = vectorizer.transform(X_test_list)

    y_train = train_df['label']
    y_test = test_df['label']

    # --- Classifier Training using all CPU cores ---
    print("\nTraining RandomForest classifier with n_jobs=-1...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train_vec, y_train)

    # --- Evaluation on Test Set ---
    print("\n--- Evaluation on Test Set ---")
    y_pred = clf.predict(X_test_vec)

    if clf.n_classes_ > 1:
        y_proba = clf.predict_proba(X_test_vec)[:, 1]
    else:
        y_proba = None

    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix (Test):")
    print(confusion_matrix(y_test, y_pred))

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"\nAccuracy (Test): {acc:.4f}")
    print(f"Macro F1   (Test): {macro_f1:.4f}")

    if y_proba is not None:
        rocauc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC    (Test): {rocauc:.4f}")

        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'AUC = {rocauc:.3f}', lw=2)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Test Data)')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()
    else:
        print("Only one class detected; ROC-AUC cannot be computed.")

    # --- Interactive Exploration ---
    print("\n--- Interactive Exploration (Test Data) ---")
    print("Enter an article index (from the test dataset index) to inspect its details, or type 'exit'.")
    print(f"Valid indices range from {test_df.index.min()} to {test_df.index.max()}")

    while True:
        user_input = input("\nIndex to inspect (or 'exit'): ")
        if user_input.strip().lower() == 'exit':
            break
        try:
            idx = int(user_input)
            if idx not in test_df.index:
                print("Index out of range for test dataset.")
                continue

            article_text = test_df.loc[idx, 'comment_text']
            true_label = test_df.loc[idx, 'label']

            tokens = dual_branch_analyzer(article_text)
            token_string = " ".join(tokens)
            vector = vectorizer.transform([token_string])
            pred_label = clf.predict(vector)[0]
            if clf.n_classes_ > 1:
                pred_proba = clf.predict_proba(vector)[0, 1]
            else:
                pred_proba = None

            print(f"\nArticle Text (first 500 chars):\n{article_text[:500]}{'...' if len(article_text) > 500 else ''}")
            print(f"True Label: {true_label} | Predicted: {pred_label}", end=" ")
            if pred_proba is not None:
                print(f"(Probability for class 1: {pred_proba:.3f})")
            else:
                print("")

            explain_prediction(article_text, vectorizer, clf)

        except ValueError:
            print("Please enter a valid integer index or 'exit'.")

    print("\nDone. Thanks for using the system!")


if __name__ == '__main__':
    main()

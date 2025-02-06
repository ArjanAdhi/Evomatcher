import os
import time
import random
import logging
import numpy as np
import pandas as pd
import spacy
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# -------------------------------------------------------------------
# LOGGING & SEED SETTING
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed set to: {seed}")


set_seed(42)

# -------------------------------------------------------------------
# Load spaCy Model Globally
# -------------------------------------------------------------------
logger.info("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")


def init_worker():
    global nlp
    nlp = spacy.load("en_core_web_sm")


# -------------------------------------------------------------------
# Data Loading & Preprocessing Functions
# -------------------------------------------------------------------
def load_factcheck_data(json_path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {json_path}...")
    df = pd.read_json(json_path, orient="records", lines=True)
    required_cols = ["verdict", "statement"]
    for col in required_cols:
        if col not in df.columns:
            raise Exception(f"Dataset must contain '{col}' column.")

    true_verdicts = {"true", "mostly-true", "half-true"}
    fake_verdicts = {"mostly-false", "false", "pants-fire"}

    def map_verdict(verdict: str):
        verdict = verdict.lower().strip()
        if verdict in true_verdicts:
            return 0
        elif verdict in fake_verdicts:
            return 1
        else:
            return np.nan

    df['label'] = df['verdict'].apply(map_verdict)
    df = df.dropna(subset=["statement", "label"])
    df['label'] = df['label'].astype(int)
    # Rename statement for consistency
    df = df.rename(columns={'statement': 'comment_text'})
    logger.info("Data loaded successfully.")
    return df


# -------------------------------------------------------------------
# Branch Feature Extraction Functions
# -------------------------------------------------------------------
def concept_mapping_score(text: str) -> list:
    """Extract noun chunks and named entities (branch 1)."""
    doc = nlp(text)
    return [chunk.text.strip() for chunk in doc.noun_chunks] + [ent.text.strip() for ent in doc.ents]


def statistical_connection_score(text: str) -> list:
    """Extract lemmatized tokens excluding stop words and punctuation (branch 2)."""
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]


def parallel_feature_extraction(texts: pd.Series, extractor_function) -> list:
    num_workers = multiprocessing.cpu_count()
    logger.info(f"Extracting features in parallel using {num_workers} workers for {extractor_function.__name__}...")
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
        results = list(executor.map(extractor_function, texts))
    return results


# -------------------------------------------------------------------
# Train Separate Branch Classifiers and Adjust Weights
# -------------------------------------------------------------------
def train_branch_classifier(features: list, labels, lowercase=True):
    # Convert list of token lists into a string for each sample.
    text_features = [" ".join(tokens) for tokens in features]
    vectorizer = CountVectorizer(lowercase=lowercase)
    X_vec = vectorizer.fit_transform(text_features)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_vec, labels)
    return vectorizer, clf


def get_branch_probabilities(vectorizer, clf, features: list):
    text_features = [" ".join(tokens) for tokens in features]
    X_vec = vectorizer.transform(text_features)
    # Get probability for class 1
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_vec)[:, 1]
    else:
        # Fallback if predict_proba is unavailable
        probs = clf.decision_function(X_vec)
    return probs


def adjust_branch_weights(val_labels, probs_branch1, probs_branch2):
    # We search over weights from 0.0 to 1.0 (for branch1) with branch2 weight = 1 - w.
    best_auc = 0.0
    best_w = 0.5
    for w in np.linspace(0, 1, 101):
        combined_probs = w * probs_branch1 + (1 - w) * probs_branch2
        auc = roc_auc_score(val_labels, combined_probs)
        if auc > best_auc:
            best_auc = auc
            best_w = w
    logger.info(f"Optimal weight for Branch1: {best_w:.2f} (Branch2 weight: {1 - best_w:.2f}), AUC: {best_auc:.4f}")
    return best_w, 1 - best_w


# -------------------------------------------------------------------
# MAIN PIPELINE: Dynamic Weight Adjustment Between Branches
# -------------------------------------------------------------------
def main():
    JSON_FILENAME = "fact.json"  # Update path as needed
    data = load_factcheck_data(JSON_FILENAME)
    logger.info("Dataset Label Distribution:\n" + str(data['label'].value_counts()))

    # Split data into training and validation sets (for weight tuning)
    train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])
    logger.info(f"Train set: {train_df.shape}  Validation set: {val_df.shape}")

    # Extract features for each branch separately on training data.
    logger.info("Extracting Branch1 features (Concept Mapping) for training data...")
    train_features_branch1 = parallel_feature_extraction(train_df['comment_text'], concept_mapping_score)
    logger.info("Extracting Branch2 features (Statistical Connection) for training data...")
    train_features_branch2 = parallel_feature_extraction(train_df['comment_text'], statistical_connection_score)

    # Train separate classifiers for each branch.
    logger.info("Training classifier for Branch1...")
    vec1, clf1 = train_branch_classifier(train_features_branch1, train_df['label'])
    logger.info("Training classifier for Branch2...")
    vec2, clf2 = train_branch_classifier(train_features_branch2, train_df['label'])

    # Now extract features for the validation set.
    logger.info("Extracting Branch1 features for validation data...")
    val_features_branch1 = parallel_feature_extraction(val_df['comment_text'], concept_mapping_score)
    logger.info("Extracting Branch2 features for validation data...")
    val_features_branch2 = parallel_feature_extraction(val_df['comment_text'], statistical_connection_score)

    # Get probability predictions for each branch on the validation set.
    probs_branch1 = get_branch_probabilities(vec1, clf1, val_features_branch1)
    probs_branch2 = get_branch_probabilities(vec2, clf2, val_features_branch2)

    # Adjust weights to combine the branch predictions.
    optimal_w1, optimal_w2 = adjust_branch_weights(val_df['label'], probs_branch1, probs_branch2)

    # Demonstrate final combined performance on the validation set.
    combined_probs = optimal_w1 * probs_branch1 + optimal_w2 * probs_branch2
    combined_auc = roc_auc_score(val_df['label'], combined_probs)
    logger.info(f"Combined Validation ROC-AUC: {combined_auc:.4f}")

    # Optionally, you might output the final combined classification report.
    # Here, we use a threshold of 0.5 on the combined probabilities.
    combined_pred = (combined_probs >= 0.5).astype(int)
    logger.info(
        "Combined Classification Report:\n" + classification_report(val_df['label'], combined_pred, zero_division=0))


if __name__ == '__main__':
    main()

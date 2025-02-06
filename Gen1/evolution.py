import os
import numpy as np
import pandas as pd
import spacy
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Load spaCy globally in the main process
nlp = spacy.load("en_core_web_sm")


def init_worker():
    global nlp
    nlp = spacy.load("en_core_web_sm")


def load_factcheck_data(json_path):
    df = pd.read_json(json_path, orient="records", lines=True)
    required_cols = ["verdict", "statement"]
    for col in required_cols:
        if col not in df.columns:
            raise Exception(f"Dataset must contain '{col}' column.")

    true_verdicts = {"true"}
    pants_fire_verdicts = {"pants-fire"}

    def map_verdict(verdict):
        verdict = verdict.lower().strip()
        if verdict in true_verdicts:
            return 0
        elif verdict in pants_fire_verdicts:
            return 1
        else:
            return np.nan

    df['label'] = df['verdict'].apply(map_verdict)
    df = df.dropna(subset=["statement", "label"])
    df['label'] = df['label'].astype(int)
    df = df.rename(columns={'statement': 'comment_text'})

    # Keep only "true" and "pants-fire" labels
    df = df[df['label'].isin([0, 1])]

    # Balance the dataset
    min_count = df['label'].value_counts().min()
    balanced_df = pd.concat([
        df[df['label'] == label].sample(n=min_count, random_state=42)
        for label in df['label'].unique()
    ]).sample(frac=1, random_state=42)

    return balanced_df


def extract_features(text):
    doc = nlp(text)
    sentences = " ".join([sent.text.strip() for sent in doc.sents])
    concepts = " ".join([chunk.text.strip() for chunk in doc.noun_chunks])
    context = " ".join([ent.text.strip() for ent in doc.ents])
    return sentences, concepts, context


def parallel_feature_extraction(texts):
    num_workers = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
        results = list(executor.map(extract_features, texts))
    return results


def train_and_evaluate(X_train, X_test, y_train, y_test, model_name):
    vectorizer = TfidfVectorizer(lowercase=True, max_features=10000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=42, n_jobs=-1)
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

    return clf, vectorizer


def plot_interactive_graph(feature_sets):
    G = nx.Graph()

    for model, features in feature_sets.items():
        for feature in features:
            G.add_edge(model, feature)

    pos = nx.spring_layout(G, seed=42, k=0.2)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='gray')))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers+text', text=node_text,
        marker=dict(size=10, color='blue', line=dict(width=2, color='black'))
    ))

    fig.update_layout(title="Interactive Feature Map", showlegend=False, hovermode='closest')
    fig.show()


def main():
    JSON_FILENAME = "fact.json"
    data = load_factcheck_data(JSON_FILENAME)
    print("\nDataset Label Distribution:")
    print(data['label'].value_counts())

    X = data['comment_text']
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print("\nExtracting concept mappings using multiprocessing...")
    processed_features = parallel_feature_extraction(X_train)

    train_sentences = [feat[0] for feat in processed_features]
    train_concepts = [feat[1] for feat in processed_features]
    train_contexts = [feat[2] for feat in processed_features]

    processed_features_test = parallel_feature_extraction(X_test)
    test_sentences = [feat[0] for feat in processed_features_test]
    test_concepts = [feat[1] for feat in processed_features_test]
    test_contexts = [feat[2] for feat in processed_features_test]

    combined_train = [s + " " + c + " " + ctx for s, c, ctx in zip(train_sentences, train_concepts, train_contexts)]
    combined_test = [s + " " + c + " " + ctx for s, c, ctx in zip(test_sentences, test_concepts, test_contexts)]

    clf, vectorizer = train_and_evaluate(combined_train, combined_test, y_train, y_test,
                                         "Adaptive Temporal Summation Model")

    feature_sets = {"Adaptive Temporal Summation Model": vectorizer.get_feature_names_out()}
    plot_interactive_graph(feature_sets)


if __name__ == '__main__':
    main()

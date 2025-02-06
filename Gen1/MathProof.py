import os
import re
import time
import random
import numpy as np
import pandas as pd
import spacy
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter, defaultdict

# ---------------------------
# Load spaCy model with full capability (dependency parser enabled)
# ---------------------------
nlp = spacy.load("en_core_web_sm")  # full model with dependency parsing, NER, etc.


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
# Emergent Concept Hierarchy Extraction
# ===========================
def extract_emergent_concepts(text):
    """
    Processes the text using spaCy and extracts emergent concept phrases.
    Here we extract noun chunks and named entities.
    Returns a list of concept strings.
    """
    doc = nlp(text)
    # Extract noun chunks
    noun_chunks = [chunk.text.strip() for chunk in doc.noun_chunks]
    # Extract named entities
    entities = [ent.text.strip() for ent in doc.ents]
    # Combine and filter out empty strings
    all_concepts = [c for c in noun_chunks + entities if c]
    return all_concepts


def build_concept_graph(texts):
    """
    Builds a co-occurrence graph of concepts.
    For each text, extract emergent concepts; for each pair of concepts, add or update an edge in the graph.
    Returns a NetworkX graph.
    """
    # Use a default dictionary to count co-occurrences
    edge_weights = defaultdict(int)
    node_counts = defaultdict(int)

    for text in texts:
        concepts = extract_emergent_concepts(text)
        # Count each concept occurrence (for node weighting later)
        for concept in set(concepts):
            node_counts[concept] += 1
        # For each unique pair of concepts in this text, add an edge weight.
        # We use sorted tuple keys to avoid duplication.
        unique_concepts = list(set(concepts))
        for i in range(len(unique_concepts)):
            for j in range(i + 1, len(unique_concepts)):
                key = tuple(sorted((unique_concepts[i], unique_concepts[j])))
                edge_weights[key] += 1

    # Create graph and add nodes and weighted edges
    G = nx.Graph()
    for concept, count in node_counts.items():
        G.add_node(concept, weight=count)
    for (c1, c2), weight in edge_weights.items():
        G.add_edge(c1, c2, weight=weight)
    return G


def draw_concept_graph(G, min_edge_weight=1):
    """
    Draws the concept graph with matplotlib.
    Only edges with weight above min_edge_weight are drawn.
    """
    # Filter edges
    edges_to_draw = [(u, v) for u, v, d in G.edges(data=True) if d.get("weight", 0) >= min_edge_weight]
    subG = G.edge_subgraph(edges_to_draw).copy()

    pos = nx.spring_layout(subG, k=0.5, seed=42)

    plt.figure(figsize=(12, 10))
    # Draw nodes with size proportional to their weight
    node_sizes = [subG.nodes[n]["weight"] * 100 for n in subG.nodes()]
    nx.draw_networkx_nodes(subG, pos, node_size=node_sizes, node_color="lightblue")

    # Draw edges with width proportional to weight
    edge_widths = [subG[u][v]["weight"] for u, v in subG.edges()]
    nx.draw_networkx_edges(subG, pos, width=edge_widths, alpha=0.6)

    # Draw labels
    nx.draw_networkx_labels(subG, pos, font_size=10)

    plt.title("Emergent Concept Co-occurrence Graph")
    plt.axis("off")
    plt.show()


# ===========================
# Main Script
# ===========================
def main():
    DATASET_FILENAME = "protests.csv"  # Ensure protests.csv is in the same directory.
    data = load_protests_data(DATASET_FILENAME)
    print("Full Dataset summary (label counts):")
    print(data['label'].value_counts())

    balanced_data = balance_dataset(data)
    print("\nBalanced Dataset summary (each class count):")
    print(balanced_data['label'].value_counts())

    # For demonstration, select all texts (or a subset) for the concept graph.
    # Here we use all balanced texts; for a quicker run, you can sample.
    texts = balanced_data['comment_text']

    # Build the emergent concept graph from the texts
    print("\nBuilding emergent concept graph from texts...")
    concept_graph = build_concept_graph(texts)

    # Print some basic stats about the graph
    print(f"Number of nodes (concepts): {concept_graph.number_of_nodes()}")
    print(f"Number of edges (co-occurrences): {concept_graph.number_of_edges()}")

    # Draw the graph (filtering out low-weight edges if desired)
    draw_concept_graph(concept_graph, min_edge_weight=1)

    # Optionally, print concept hierarchies for a few sample texts
    sample_texts = texts.sample(n=5, random_state=42)
    print("\n--- Emergent Concept Hierarchies for Sample Texts ---")
    for i, text in enumerate(sample_texts, start=1):
        print(f"\nSample Text {i}:")
        print(text)
        print("Concept Hierarchy:")
        counts = Counter(extract_emergent_concepts(text))
        for concept, count in counts.most_common():
            print(f"  {concept}: {count}")
        print("-" * 50)


if __name__ == '__main__':
    main()

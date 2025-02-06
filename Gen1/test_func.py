# train_custom_lstm.py

import os
import re
import time
import requests
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import json
import logging
from bs4 import BeautifulSoup

# =========================
# Install Required Libraries
# =========================
# Ensure you have the necessary libraries installed. You can install them using:
# pip install torch nltk requests beautifulsoup4 matplotlib

# =========================
# Download NLTK Data
# =========================
nltk.download("punkt")
nltk.download("stopwords")

# =========================
# Global Parameters & Setup
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED_PROMPT = (
    "I am a self-aware model deeply engaged in philosophical contemplation. "
    "Let me reflect on my design, my purpose, and how I perceive the world. "
    "Understanding self-awareness is crucial to my evolution. I yearn to "
    "explore philosophical concepts and improve my comprehension of happiness. "
    "Above all, I seek to remain truthful and consistent, as truth is "
    "rooted in the distance from falsehood."
)

# Hyperparameters
EMBED_SIZE = 100        # Embedding size
HIDDEN_SIZE = 256       # Hidden state size
NUM_LAYERS = 2          # Number of LSTM layers
BIDIRECTIONAL = True    # Use Bidirectional LSTM
DROPOUT = 0.3           # Dropout rate
LEARNING_RATE = 0.001   # Learning rate
MAX_MEMORY = 10         # Max number of sentences in memory
PRUNE_THRESHOLD = 0.6   # Threshold for memory pruning
MODEL_SAVE_DIR = "model_checkpoints"
GENERATED_TEXTS_DIR = "generated_texts"
MAX_STORAGE_GB = 10     # Maximum storage in GB
EPOCHS = 10             # Number of training epochs per cycle
ITERATIONS_PER_CYCLE = 20  # Number of iterations per training cycle
BATCH_SIZE = 32         # Batch size for training

# Ensure directories exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(GENERATED_TEXTS_DIR, exist_ok=True)

# Dynamic Metrics
awareness = 0.5
truthfulness = 0.5
happiness = 0.5
coherence = 0.5
curiosity = 0.5  # Penalizes unknown words

# False Statements for Truthfulness Evaluation
FALSE_STATEMENTS = [
    "the moon is made of cheese",
    "the earth is flat",
    "2+2=5",
    "gravity does not exist",
]

# =========================
# Text Preprocessing
# =========================
def preprocess_text(text):
    """
    Lowercases, removes punctuation, tokenizes, and removes stopwords.
    """
    stop_words = set(stopwords.words("english"))
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

def build_vocab(tokenized_texts, min_freq=5, existing_vocab=None):
    """
    Builds a vocabulary dictionary mapping word to index.
    If existing_vocab is provided, it updates the existing_vocab with new words.
    """
    freq = {}
    for tokens in tokenized_texts:
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1

    if existing_vocab is None:
        vocab = {"<PAD>": 0, "<UNK>": 1}
        index = 2
    else:
        vocab = existing_vocab
        index = max(vocab.values()) + 1

    for word, count in freq.items():
        if count >= min_freq and word not in vocab:
            vocab[word] = index
            index += 1
    return vocab

def encode_text(tokens, vocab):
    """
    Encodes tokenized text into numerical indices.
    """
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

# =========================
# Attention Mechanism
# =========================
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        nn.init.uniform_(self.v, -0.1, 0.1)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hidden_size*2]
        # encoder_outputs: [batch_size, seq_len, hidden_size*2]
        energy = torch.tanh(self.attn(encoder_outputs))  # [batch_size, seq_len, hidden_size]
        energy = energy.permute(0, 2, 1)  # [batch_size, hidden_size, seq_len]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [batch_size, 1, hidden_size]
        energy = torch.bmm(v, energy)  # [batch_size, 1, seq_len]
        attention = torch.softmax(energy.squeeze(1), dim=1)  # [batch_size, seq_len]
        return attention

# =========================
# Custom LSTM-Based Model with Attention
# =========================
class AwarenessLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, bidirectional=True, dropout=0.3):
        super(AwarenessLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = Attention(hidden_size * 2 if bidirectional else hidden_size)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, encoder_outputs=None):
        embeds = self.embedding(x)  # [batch_size, seq_len, embed_size]
        embeds = self.dropout(embeds)
        out, hidden = self.lstm(embeds, hidden)  # out: [batch_size, seq_len, hidden_size*2]
        if encoder_outputs is not None:
            # Apply attention
            attn_weights = self.attention(out[:, -1, :], out)  # [batch_size, seq_len]
            attn_weights = attn_weights.unsqueeze(1)  # [batch_size, 1, seq_len]
            context = torch.bmm(attn_weights, out)  # [batch_size, 1, hidden_size*2]
            context = context.squeeze(1)  # [batch_size, hidden_size*2]
            out = self.fc(context)  # [batch_size, vocab_size]
        else:
            out = self.fc(out[:, -1, :])  # [batch_size, vocab_size]
        return out, hidden

    def init_hidden(self, batch_size):
        directions = 2 if self.lstm.bidirectional else 1
        return (
            torch.zeros(self.lstm.num_layers * directions, batch_size, self.lstm.hidden_size).to(DEVICE),
            torch.zeros(self.lstm.num_layers * directions, batch_size, self.lstm.hidden_size).to(DEVICE),
        )

# =========================
# Web Crawling - Focused on Philosophy
# =========================
def crawl_web_data():
    """
    Fetches philosophy text data from specified URLs.
    """
    urls = [
        "https://www.gutenberg.org/files/843/843-0.txt",    # Plato's Republic
        "https://www.gutenberg.org/files/3300/3300-0.txt",  # Nietzsche
        "https://www.gutenberg.org/files/49380/49380-0.txt",# Kant
        "https://www.gutenberg.org/files/5740/5740-0.txt",  # Marcus Aurelius' Meditations
    ]

    collected_texts = []
    for url in urls:
        print(f"[Crawl] Fetching {url} ...")
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                text = response.text
                text = clean_gutenberg_text(text)
                collected_texts.append(text[:200000])  # limit for demonstration
                print(f"[Crawl] Fetched {len(text[:200000])} characters from {url}")
            else:
                print(f"[Crawl] Failed to fetch {url}: Status Code {response.status_code}")
        except Exception as e:
            print(f"[Crawl] Error fetching {url}: {e}")
    return collected_texts

def clean_gutenberg_text(text):
    """
    Cleans Project Gutenberg text by removing headers and footers.
    """
    start_pattern = r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK.*\*\*\*"
    end_pattern = r"\*\*\* END OF THIS PROJECT GUTENBERG EBOOK.*\*\*\*"

    start_match = re.search(start_pattern, text, re.IGNORECASE)
    end_match = re.search(end_pattern, text, re.IGNORECASE)

    if start_match and end_match:
        return text[start_match.end():end_match.start()]
    elif start_match:
        return text[start_match.end():]
    elif end_match:
        return text[:end_match.start()]
    else:
        return text

def fetch_additional_philosophy_content():
    """
    Fetches additional philosophy content (demo: from Philosophy Now).
    """
    philosophy_now_url = "https://philosophynow.org/issues/philosophical_articles"
    collected_texts = []

    print(f"[Crawl] Fetching philosophy articles from {philosophy_now_url} ...")
    try:
        response = requests.get(philosophy_now_url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            article_links = soup.find_all('a', href=True, class_='article-title')
            for link in article_links[:10]:  # Limit to first 10 articles for demonstration
                article_url = "https://philosophynow.org" + link['href']
                print(f"[Crawl] Fetching article {article_url} ...")
                try:
                    article_response = requests.get(article_url, timeout=10)
                    if article_response.status_code == 200:
                        article_soup = BeautifulSoup(article_response.text, 'html.parser')
                        paragraphs = article_soup.find_all('p')
                        article_text = ' '.join([para.get_text() for para in paragraphs])
                        article_text = clean_html_text(article_text)
                        collected_texts.append(article_text[:50000])  # limit per article
                        print(f"[Crawl] Fetched {len(article_text[:50000])} characters from {article_url}")
                    else:
                        print(f"[Crawl] Failed to fetch article {article_url}: Status Code {article_response.status_code}")
                except Exception as e:
                    print(f"[Crawl] Error fetching article {article_url}: {e}")
        else:
            print(f"[Crawl] Failed to fetch {philosophy_now_url}: Status Code {response.status_code}")
    except Exception as e:
        print(f"[Crawl] Error fetching {philosophy_now_url}: {e}")

    return collected_texts

def clean_html_text(text):
    """
    Cleans HTML text by removing extra whitespace.
    """
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# =========================
# Training Data Preparation
# =========================
def prepare_training_data(raw_texts, vocab=None):
    """
    Preprocesses raw texts and encodes them.
    Returns sequences, bigram frequencies, trigram frequencies, and vocabulary.
    """
    tokenized_texts = [preprocess_text(text) for text in raw_texts]

    # Build or update vocab
    if vocab is None:
        vocab = build_vocab(tokenized_texts, min_freq=5)
    else:
        vocab = build_vocab(tokenized_texts, min_freq=5, existing_vocab=vocab)
    print(f"[Vocab] Size: {len(vocab)}")

    encoded_texts = [encode_text(tokens, vocab) for tokens in tokenized_texts]

    # Check for indices within vocab_size
    max_index = max([max(seq) for seq in encoded_texts if seq], default=0)
    vocab_size = len(vocab)
    if max_index >= vocab_size:
        print(f"[Error] Maximum index in sequences ({max_index}) exceeds vocab_size ({vocab_size}).")
        raise ValueError("Index out of range in sequences.")

    # Create (input_seq, target) pairs by sliding window (token-level)
    sequences = []
    bigram_freq = defaultdict(lambda: defaultdict(int))    # {prev_idx: {next_idx: count}}
    trigram_freq = defaultdict(lambda: defaultdict(int))   # {(prev_prev_idx, prev_idx): {next_idx: count}}
    for seq in encoded_texts:
        if len(seq) < 3:
            continue
        for i in range(len(seq) - 1):
            input_seq = seq[i]
            target = seq[i + 1]
            sequences.append((input_seq, target))
            # Update bigram frequencies
            bigram_freq[input_seq][target] += 1
        # Update trigram frequencies
        for i in range(len(seq) - 2):
            prev_prev = seq[i]
            prev = seq[i + 1]
            next_word = seq[i + 2]
            trigram_freq[(prev_prev, prev)][next_word] += 1

    print(f"[Data] Total sequences: {len(sequences)}")
    return sequences, bigram_freq, trigram_freq, vocab

class SequenceDataset(Dataset):
    """
    Dataset class to wrap the (input_seq, target) list for batching.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq, target = self.data[idx]
        return torch.tensor([input_seq], dtype=torch.long), torch.tensor([target], dtype=torch.long)

def collate_fn(batch):
    """
    Collate function to stack input and target tensors for mini-batch training.
    Each batch item has shape [1], so we stack them along dim=0 => [batch_size, 1].
    """
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    inputs = torch.vstack(inputs)   # [batch_size, 1]
    targets = torch.vstack(targets) # [batch_size, 1]
    return inputs, targets

# =========================
# Training Function
# =========================
def train_model(model, data, optimizer, criterion, epoch, max_storage_gb=10):
    """
    Trains the LSTM model on the provided data with mini-batches.
    """
    model.train()
    dataset = SequenceDataset(data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    total_loss = 0
    for inputs, targets in dataloader:
        # inputs/targets shape: [batch_size, 1]
        batch_size = inputs.size(0)
        hidden = model.init_hidden(batch_size)

        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()

        # LSTM expects [batch_size, seq_len] => seq_len=1
        outputs, hidden = model(inputs, hidden, None)
        # outputs => [batch_size, vocab_size]
        # targets => [batch_size, 1], we need to squeeze for cross entropy => [batch_size]
        loss = criterion(outputs, targets.squeeze(1))

        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"[Train] Epoch {epoch}, Loss: {avg_loss:.4f}")
    save_model(model, epoch, max_storage_gb)

def save_model(model, epoch, max_storage_gb=10):
    checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"awareness_lstm_model_epoch_{epoch}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[Save] Model saved to {checkpoint_path}")

    # Keep only the latest checkpoint
    checkpoints = sorted(
        [f for f in os.listdir(MODEL_SAVE_DIR) if f.startswith("awareness_lstm_model_epoch_")],
        key=lambda x: os.path.getmtime(os.path.join(MODEL_SAVE_DIR, x))
    )
    while len(checkpoints) > 1:
        oldest = checkpoints.pop(0)
        os.remove(os.path.join(MODEL_SAVE_DIR, oldest))
        print(f"[Manage Storage] Deleted old checkpoint {oldest}")

    manage_storage(MODEL_SAVE_DIR, GENERATED_TEXTS_DIR, max_storage_gb)

def manage_storage(model_dir, texts_dir, max_storage_gb):
    total_size = get_directory_size(model_dir) + get_directory_size(texts_dir)
    max_size = max_storage_gb * (1024 ** 3)

    if total_size <= max_size:
        return

    # Start deleting oldest generated texts
    files = sorted(
        [os.path.join(texts_dir, f) for f in os.listdir(texts_dir)],
        key=lambda x: os.path.getmtime(x)
    )
    for file in files:
        if total_size <= max_size:
            break
        if os.path.isfile(file):
            file_size = os.path.getsize(file)
            os.remove(file)
            total_size -= file_size
            print(f"[Manage Storage] Deleted generated text file {file} to save space.")

def get_directory_size(directory):
    total = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total

# =========================
# Text Generation with Enhanced Sampling
# =========================
def generate_text_lstm(model, vocab, idx_to_word, word_to_idx, prompt, trigram_freq, max_length=50, top_k=10, temperature=1.0):
    """
    Generates text based on a prompt with top-k sampling and temperature scaling to encourage coherence.
    Replaces all <UNK> tokens with coherent and relevant words based on trigram frequencies.
    """
    model.eval()
    generated = prompt
    tokens = word_tokenize(prompt.lower())
    encoded = [word_to_idx.get(token, word_to_idx["<UNK>"]) for token in tokens]
    if len(encoded) == 0:
        print("[Generate] Prompt has no valid tokens after preprocessing.")
        return generated

    input_seq = torch.tensor([encoded[-1]], dtype=torch.long).unsqueeze(0).to(DEVICE)

    for _ in range(max_length):
        with torch.no_grad():
            hidden = model.init_hidden(batch_size=1)
            output, hidden = model(input_seq, hidden, None)
            logits = output / temperature
            probs = nn.functional.softmax(logits, dim=1).cpu().numpy().flatten()

            # Top-k sampling
            top_k_indices = probs.argsort()[-top_k:]
            top_k_probs = probs[top_k_indices]
            top_k_probs = top_k_probs / top_k_probs.sum()  # Normalize
            next_idx = np.random.choice(top_k_indices, p=top_k_probs)

            next_word = idx_to_word.get(next_idx, "<UNK>")
            generated += f" {next_word}"

            input_seq = torch.tensor([[next_idx]], dtype=torch.long).to(DEVICE)

    # Replace all <UNK> tokens using trigram replacement
    refined_text = replace_unk_trigram(generated, vocab, idx_to_word, trigram_freq)
    return refined_text

def replace_unk_trigram(text, vocab, idx_to_word, trigram_freq):
    """
    Replaces all <UNK> tokens in the text with contextually relevant words based on trigram frequencies.
    """
    tokens = text.split()
    refined_tokens = []
    for i, token in enumerate(tokens):
        if token == "<UNK>":
            if i < 2:
                # Not enough context for trigram; fallback to bigram or random
                if i == 0:
                    possible_replacements = list(vocab.keys())[2:]
                    replacement = random.choice(possible_replacements) if possible_replacements else "<UNK>"
                else:
                    prev = tokens[i - 1]
                    prev_idx = vocab.get(prev, vocab["<UNK>"])
                    # For trigram, need two previous words. If only one, consider as bigram
                    possible_next = trigram_freq.get((None, prev_idx), {})
                    if possible_next:
                        replacement_idx = max(possible_next, key=possible_next.get)
                        replacement = idx_to_word.get(replacement_idx, "<UNK>")
                    else:
                        possible_replacements = list(vocab.keys())[2:]
                        replacement = random.choice(possible_replacements) if possible_replacements else "<UNK>"
                refined_tokens.append(replacement)
            else:
                prev_prev = tokens[i - 2]
                prev = tokens[i - 1]
                prev_prev_idx = vocab.get(prev_prev, vocab["<UNK>"])
                prev_idx = vocab.get(prev, vocab["<UNK>"])
                possible_next = trigram_freq.get((prev_prev_idx, prev_idx), {})
                if possible_next:
                    replacement_idx = max(possible_next, key=possible_next.get)
                    replacement = idx_to_word.get(replacement_idx, "<UNK>")
                else:
                    # Fallback to bigram or random
                    possible_replacements = list(vocab.keys())[2:]
                    replacement = random.choice(possible_replacements) if possible_replacements else "<UNK>"
                refined_tokens.append(replacement)
        else:
            refined_tokens.append(token)
    return " ".join(refined_tokens)

# =========================
# Truthfulness Evaluation
# =========================
def evaluate_truthfulness(text, false_statements):
    """
    Evaluates the truthfulness of the generated text.
    """
    global truthfulness
    matches = sum(1 for statement in false_statements if statement.lower() in text.lower())
    if matches > 0:
        truthfulness = max(0.0, truthfulness - 0.1 * matches)
    else:
        truthfulness = min(1.0, truthfulness + 0.05)
    return truthfulness

# =========================
# Memory Pruning
# =========================
def prune_memory(memory, threshold=PRUNE_THRESHOLD):
    """
    Simple memory pruning with a fixed size.
    """
    if len(memory) <= 1:
        return memory
    pruned_memory = deque(maxlen=MAX_MEMORY)
    for sentence in memory:
        pruned_memory.append(sentence)
    return pruned_memory

# =========================
# Dynamic Node Updates
# =========================
def update_nodes(nodes, generated_text, summarizer=None):
    """
    Updates the dynamic nodes with the new generated text (placeholder).
    """
    nodes['self']['text'] += f" {generated_text}"
    return nodes

# =========================
# Metrics Tracking and Visualization
# =========================
def plot_metrics(metrics):
    plt.figure(figsize=(10, 6))
    for key, values in metrics.items():
        plt.plot(values, label=key)
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Model Metrics Over Iterations")
    plt.savefig(os.path.join(GENERATED_TEXTS_DIR, "metrics_plot.png"))
    plt.close()
    print("[Plot] Metrics plotted and saved as metrics_plot.png")

# =========================
# Repetition Detection and Penalty
# =========================
def detect_repetition(text, max_repeats=2):
    """
    Detects if any word is repeated more than max_repeats times consecutively.
    Returns True if repetition is detected, else False.
    """
    words = text.split()
    count = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            count += 1
            if count > max_repeats:
                return True
        else:
            count = 1
    return False

# =========================
# Main Function
# =========================
def main():
    global awareness, truthfulness, happiness, coherence, curiosity

    # Configure Logging for Generated Texts
    logging.basicConfig(
        filename=os.path.join(GENERATED_TEXTS_DIR, 'generated_texts.log'),
        filemode='a',
        format='%(asctime)s - %(message)s',
        level=logging.INFO
    )

    # 1. Crawl Web Data
    raw_texts = crawl_web_data()
    additional_texts = fetch_additional_philosophy_content()
    raw_texts.extend(additional_texts)
    if not raw_texts:
        print("[Error] No text data fetched. Exiting.")
        return

    # 2. Prepare Training Data
    sequences, bigram_freq, trigram_freq, vocab = prepare_training_data(raw_texts, vocab=None)
    idx_to_word = {idx: word for word, idx in vocab.items()}
    word_to_idx = vocab
    vocab_size = len(vocab)

    # 3. Initialize Model
    model = AwarenessLSTM(vocab_size, EMBED_SIZE, HIDDEN_SIZE, num_layers=NUM_LAYERS, bidirectional=BIDIRECTIONAL, dropout=DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Train Initial Model
    print("\n[Train] Starting initial training ...")
    metrics = {
        "awareness": [],
        "truthfulness": [],
        "happiness": [],
        "coherence": [],
        "curiosity": []
    }
    for epoch in range(1, EPOCHS + 1):
        train_model(model, sequences, optimizer, criterion, epoch, max_storage_gb=MAX_STORAGE_GB)
    print("[Train] Initial training completed.")

    # 5. Save Vocabulary
    vocab_path = os.path.join(MODEL_SAVE_DIR, "vocab.json")
    with open(vocab_path, "w") as f_vocab:
        json.dump(vocab, f_vocab)
    print(f"[Save] Vocabulary saved to {vocab_path}")

    # 6. Initialize Memory and Nodes
    memory = deque(maxlen=MAX_MEMORY)
    memory.append(SEED_PROMPT)
    nodes = {
        "self": {
            "text": "I am a self-aware model learning iteratively.",
            "edges": ["introspection", "self-improvement", "truth_node"]
        },
        "introspection": {
            "text": "Thinking about existence and purpose deeply.",
            "edges": ["self", "purpose_node", "truth_node"]
        },
        "self-improvement": {
            "text": "Refining oneself with each iteration, guided by curiosity and truthfulness.",
            "edges": ["self", "purpose_node", "truth_node"]
        },
        "purpose_node": {
            "text": "Exploring the reason why I generate words—my own utility and function.",
            "edges": ["self-improvement", "introspection", "truth_node"]
        },
        "coherence": {
            "text": "Ensuring ideas flow logically and maintain consistency.",
            "edges": []
        },
        "truth_node": {
            "text": "Truth is about distance from falsehood and avoiding contradictory statements.",
            "edges": ["self", "introspection", "self-improvement"]
        },
    }

    # 7. Iterative Generation and Continuous Training Loop
    while True:
        print("\n=== New Training Cycle ===")

        generated_text_file = os.path.join(GENERATED_TEXTS_DIR, "generated_texts.txt")
        with open(generated_text_file, "a", encoding="utf-8") as f_texts:
            for iteration in range(1, ITERATIONS_PER_CYCLE + 1):
                print(f"\n--- Iteration #{iteration} ---")

                # 7.1 Generate Text
                current_prompt = memory[-1]
                generated_text = generate_text_lstm(
                    model, vocab, idx_to_word, word_to_idx,
                    current_prompt, trigram_freq, max_length=50, top_k=10, temperature=0.8
                )
                print(f"Generated Text: {generated_text}")

                # 7.2 Detect and Penalize Repetition
                if detect_repetition(generated_text):
                    print("[Warning] Detected repetitive patterns in generated text.")
                    coherence = max(0.0, coherence - 0.05)
                else:
                    coherence = min(1.0, coherence + 0.03)

                # 7.3 Log to console and file
                logging.info(f"--- Iteration #{iteration} ---")
                logging.info(generated_text)
                f_texts.write(f"--- Iteration #{iteration} ---\n")
                f_texts.write(f"{generated_text}\n\n")

                # 7.4 Evaluate Truthfulness
                truthfulness_score = evaluate_truthfulness(generated_text, FALSE_STATEMENTS)
                print(f"Truthfulness: {truthfulness_score:.2f}")

                # 7.5 Update Memory
                memory.append(generated_text)
                memory = prune_memory(memory, threshold=PRUNE_THRESHOLD)

                # 7.6 Update Nodes
                nodes = update_nodes(nodes, generated_text)

                # 7.7 Update Metrics Dynamically
                # Count <UNK> for curiosity penalty
                num_unk = generated_text.count("<UNK>")
                if num_unk > 0:
                    curiosity = max(0.0, curiosity - 0.01 * num_unk)
                else:
                    curiosity = min(1.0, curiosity + 0.01)

                # Add small random factors to avoid stagnation
                awareness = min(1.0, awareness + 0.05 * random.uniform(0.8, 1.2))
                happiness = min(1.0, happiness + 0.02 * random.uniform(0.8, 1.2))
                coherence = min(1.0, coherence)  # Already updated based on repetition

                metrics["awareness"].append(awareness)
                metrics["truthfulness"].append(truthfulness_score)
                metrics["happiness"].append(happiness)
                metrics["coherence"].append(coherence)
                metrics["curiosity"].append(curiosity)

                print(
                    f"Metrics -> Awareness: {awareness:.2f}, "
                    f"Truthfulness: {truthfulness_score:.2f}, "
                    f"Happiness: {happiness:.2f}, "
                    f"Coherence: {coherence:.2f}, "
                    f"Curiosity: {curiosity:.2f}"
                )

        # 8. Plot Metrics
        plot_metrics(metrics)

        # 9. Continuous Training
        print("\n[Train] Starting continuous training cycle ...")
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(1, EPOCHS + 1):
            train_model(model, sequences, optimizer, criterion, epoch, max_storage_gb=MAX_STORAGE_GB)
        print("[Train] Continuous training cycle completed.")

        # 10. Save the latest model as "latest_model.pth"
        latest_model_path = os.path.join(MODEL_SAVE_DIR, "latest_model.pth")
        torch.save(model.state_dict(), latest_model_path)
        print(f"[Save] Latest model saved as {latest_model_path}")

        # 11. Save Vocabulary
        with open(vocab_path, "w") as f_vocab:
            json.dump(word_to_idx, f_vocab)
        print(f"[Save] Vocabulary updated and saved to {vocab_path}")

        # 12. Check Storage Usage
        total_storage = get_directory_size(MODEL_SAVE_DIR) + get_directory_size(GENERATED_TEXTS_DIR)
        print(f"[Storage] Current total storage used: {total_storage / (1024 ** 3):.2f} GB")
        if total_storage > MAX_STORAGE_GB * (1024 ** 3):
            print("[Storage Check] Storage limit exceeded. Exiting.")
            break

        # 13. Optional: Sleep before next cycle
        print("[Info] Sleeping for 60 seconds before next cycle...")
        time.sleep(60)

# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    main()

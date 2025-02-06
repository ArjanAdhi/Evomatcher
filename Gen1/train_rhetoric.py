import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import tempfile
import logging
import wikipedia
import numpy as np
import networkx as nx
import shutil
import threading
from datetime import datetime

# Force legacy (non-safetensors) saving globally.
os.environ["USE_SAFE_TENSORS"] = "0"

from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    TextDataset,
    DataCollatorForLanguageModeling,
    pipeline,
)
from sentence_transformers import SentenceTransformer, util

MAX_GEN_WORDS = 200

# -------------------------------------------------------------------
# LOGGING & SEED SETTING
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info(f"Random seed set to: {seed}")

def reset_random_seed():
    set_seed(random.randint(0, 1000000))

# -------------------------------------------------------------------
# CUSTOM MODEL CONFIGURATION & ARCHITECTURE (From Scratch)
# -------------------------------------------------------------------
class CustomConfig(PretrainedConfig):
    model_type = "customlm"
    def __init__(self, vocab_size=50257, d_model=256, num_layers=4, num_heads=4, dropout=0.1,
                 n_positions=512, n_ctx=512, bos_token_id=50256, eos_token_id=50256, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

class CustomLMModel(PreTrainedModel):
    config_class = CustomConfig
    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, config.n_positions, config.d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.num_heads, dropout=config.dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.init_weights()
    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids) + self.positional_encoding[:, :input_ids.size(1), :]
        x = x.transpose(0, 1)  # (T, B, d_model)
        hidden_states = self.encoder(x)
        hidden_states = hidden_states.transpose(0, 1)  # (B, T, d_model)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return {"loss": loss, "logits": logits}

# Two branch configurations.
class LexicalConfig(CustomConfig):
    def __init__(self, **kwargs):
        super().__init__(d_model=256, num_layers=4, num_heads=4, n_positions=512, n_ctx=512, **kwargs)

class StatisticalConfig(CustomConfig):
    def __init__(self, **kwargs):
        super().__init__(d_model=512, num_layers=6, num_heads=8, n_positions=1024, n_ctx=1024, **kwargs)

# Simple branch modules.
class LexicalBranch(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(LexicalBranch, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        output, (hn, _) = self.lstm(x)
        return hn.squeeze(0)

class StatisticalBranch(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(StatisticalBranch, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        avg_emb = x.mean(dim=1)
        h = F.relu(self.fc1(avg_emb))
        h = F.relu(self.fc2(h))
        return h

class CombinedModel(nn.Module):
    def __init__(self, vocab_size, lexical_config, statistical_config, combined_hidden_dim):
        super(CombinedModel, self).__init__()
        self.lexical_branch = LexicalBranch(vocab_size, lexical_config.d_model, lexical_config.d_model)
        self.statistical_branch = StatisticalBranch(vocab_size, statistical_config.d_model, statistical_config.d_model)
        combined_input_dim = lexical_config.d_model + statistical_config.d_model
        self.combined_fc = nn.Linear(combined_input_dim, combined_hidden_dim)
        self.output_fc = nn.Linear(combined_hidden_dim, vocab_size)
        self.neuromodulator = nn.Parameter(torch.tensor(1.0))
    def forward(self, input_ids):
        lex_out = self.lexical_branch(input_ids)
        stat_out = self.statistical_branch(input_ids)
        combined = torch.cat([lex_out, stat_out], dim=1) * self.neuromodulator
        hidden = F.relu(self.combined_fc(combined))
        logits = self.output_fc(hidden)
        return logits

# -------------------------------------------------------------------
# DEVICE & OUTPUT DIRECTORY SETUP
# -------------------------------------------------------------------
LEXICAL_DEVICE = torch.device("cpu")
STATISTICAL_DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

logger.info(f"Lexical branch will use device: {LEXICAL_DEVICE}")
logger.info(f"Statistical branch will use device: {STATISTICAL_DEVICE}")

LEXICAL_OUTPUT_DIR = "Lexical_Model"
STATISTICAL_OUTPUT_DIR = "Statistical_Model"
COMBINED_OUTPUT_DIR = "Combined_Model"
SAVE_TOTAL_LIMIT = 3

# -------------------------------------------------------------------
# HELPER FUNCTIONS: REMOVE REPETITIONS & TRUNCATE TEXT
# -------------------------------------------------------------------
def remove_repetitions(text: str) -> str:
    sentences = []
    seen = set()
    for sentence in text.split('.'):
        sentence = sentence.strip()
        if sentence and sentence not in seen:
            seen.add(sentence)
            sentences.append(sentence)
    return '. '.join(sentences) + '.'

def truncate_text(text: str, max_words=MAX_GEN_WORDS) -> str:
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text

# -------------------------------------------------------------------
# SAFE SAVE FUNCTION
# -------------------------------------------------------------------
def safe_save_model(trainer: Trainer, output_dir: str, max_retries=5, delay=5):
    for attempt in range(max_retries):
        try:
            temp_dir = output_dir + "_tmp"
            # Remove any existing temporary directory.
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
            # Save using legacy (non-safe) serialization.
            trainer.model.save_pretrained(temp_dir, safe_serialization=False)
            # Remove the existing output directory to avoid file locks.
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.rename(temp_dir, output_dir)
            logger.info(f"✅ Model saved to '{output_dir}'")
            return
        except Exception as e:
            logger.warning(f"Save attempt {attempt+1}/{max_retries} failed: {e}")
            time.sleep(delay)
    logger.error("❌ Failed to save model after multiple attempts.")

# -------------------------------------------------------------------
# COHERENCY SCORE FUNCTION
# -------------------------------------------------------------------
def calculate_coherency_score(generated_text: str, reference_texts: list, threshold=0.4):
    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        gen_embedding = embedder.encode(generated_text, convert_to_tensor=True)
        ref_embeddings = embedder.encode(reference_texts, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(gen_embedding, ref_embeddings).cpu().numpy()
        avg_similarity = np.mean(similarities)
        logger.info(f"[COHERENCY SCORE] {avg_similarity:.3f}")
        return (avg_similarity >= threshold), avg_similarity
    except Exception as e:
        logger.exception("Error calculating coherency score.")
        return (False, 0.0)

# -------------------------------------------------------------------
# HOMEOSTATIC STATE & BRAIN-INSPIRED VARIABLES
# -------------------------------------------------------------------
homeostatic_state = {
    "conscious": {"context": "immediate emergent concepts", "weight": 0.6},
    "subconscious": {"context": "accumulated long-term knowledge", "weight": 0.4},
}
synaptic_plasticity = 0.5

def update_homeostatic_state(new_sample: str):
    homeostatic_state["conscious"]["context"] = new_sample
    homeostatic_state["subconscious"]["context"] += " " + new_sample[-500:]
    new_weight = random.uniform(0.4, 0.8)
    homeostatic_state["conscious"]["weight"] = new_weight
    homeostatic_state["subconscious"]["weight"] = 1.0 - new_weight
    neurotransmitter_level = random.uniform(0.8, 1.2)
    logger.info(f"Nodes updated: Conscious weight = {new_weight:.2f}, Subconscious weight = {1.0 - new_weight:.2f}, Neurotransmitter level = {neurotransmitter_level:.2f}")

def get_combined_context() -> str:
    c = homeostatic_state["conscious"]
    s = homeostatic_state["subconscious"]
    return (f"Conscious: {c['context']}\nSubconscious: {s['context']}\n"
            f"Weights: conscious={c['weight']:.2f}, subconscious={s['weight']:.2f}")

# -------------------------------------------------------------------
# WIKIPEDIA DATA RETRIEVAL & CONCEPT MAPPING
# -------------------------------------------------------------------
SEARCH_TERMS = ["Communication", "AI", "Psychology", "Neural Networks", "Language Processing"]

def fetch_wikipedia_summary(title: str):
    try:
        return wikipedia.summary(title, sentences=5)
    except Exception as e:
        logger.warning(f"Skipping '{title}' due to error: {e}")
        return None

def get_wikipedia_data(num_articles=10) -> dict:
    summaries = {}
    for term in SEARCH_TERMS:
        results = wikipedia.search(term, results=num_articles)
        for title in results:
            summary = fetch_wikipedia_summary(title)
            if summary:
                summaries[title] = summary
    logger.info(f"Retrieved {len(summaries)} Wikipedia summaries.")
    return summaries

def encode_and_find_connections(summaries: dict, threshold=0.3) -> list:
    texts = [f"{title}: {summary}" for title, summary in summaries.items()]
    if not texts:
        logger.warning("No text to encode.")
        return []
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    embeddings = embedder.encode(texts, convert_to_tensor=True)
    sim_matrix = util.pytorch_cos_sim(embeddings, embeddings).cpu().numpy()
    connections = []
    titles = list(summaries.keys())
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            score = sim_matrix[i][j]
            if score > threshold:
                connections.append((titles[i], titles[j], float(score)))
    connections.sort(key=lambda x: x[2], reverse=True)
    connections = connections[:40]
    logger.info(f"Found {len(connections)} connections above threshold {threshold}.")
    return connections

def build_graph(connections: list):
    G = nx.Graph()
    for a, b, score in connections:
        weight_factor = random.uniform(0.8, 1.2)
        if not G.has_node(a):
            G.add_node(a, node_weight=random.uniform(0.5, 1.5))
        if not G.has_node(b):
            G.add_node(b, node_weight=random.uniform(0.5, 1.5))
        G.add_edge(a, b, weight=score * weight_factor)
    for node in G.nodes():
        G.nodes[node]['node_weight'] *= random.uniform(0.9, 1.1)
    return G

def build_concept_hierarchy(graph) -> (str, dict):
    if len(graph.edges) == 0:
        logger.error("Graph has no edges; cannot build hierarchy.")
        return None, {}
    centrality = {}
    for node in graph.nodes():
        deg = graph.degree(node, weight='weight')
        w = graph.nodes[node].get('node_weight', 1)
        centrality[node] = deg * w
    root = max(centrality, key=centrality.get)
    logger.info(f"Selected '{root}' as the hierarchy root based on dynamic node weights.")
    hierarchy = {}
    visited = set()
    queue = [(root, None)]
    while queue:
        current, parent = queue.pop(0)
        if parent is not None:
            hierarchy.setdefault(parent, []).append(current)
        visited.add(current)
        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                queue.append((neighbor, current))
    return root, hierarchy

def format_hierarchy(root: str, hierarchy: dict, level=0) -> str:
    indent = "  " * level
    result = f"{indent}- {root}\n"
    for child in hierarchy.get(root, []):
        result += format_hierarchy(child, hierarchy, level+1)
    return result

def generate_initial_prompt(root: str, hierarchy: dict, summaries: dict) -> str:
    hierarchy_str = format_hierarchy(root, hierarchy)
    snippet_dict = {topic: (summary.split('. ')[0].strip() if summary else "") for topic, summary in summaries.items()}
    def traverse(node, path):
        cur_path = path + [node]
        if node not in hierarchy or not hierarchy[node]:
            return [cur_path]
        paths = []
        for child in hierarchy[node]:
            paths.extend(traverse(child, cur_path))
        return paths
    all_paths = traverse(root, [])
    longest_path = max(all_paths, key=lambda p: len(p))
    if len(longest_path) < 3:
        siblings = hierarchy.get(root, [])
        if siblings and len(siblings) >= 2:
            longest_path = [root, siblings[0], siblings[1]]
    trace_str = " -> ".join([f"'{topic}' (snippet: \"{snippet_dict.get(topic, '')}\")" for topic in longest_path])
    comb_context = get_combined_context()
    prompt = (
        f"{comb_context}\n\n"
        f"=== Concept Hierarchy ===\n{hierarchy_str}\n\n"
        f"Trace-back chain: {trace_str}\n\n"
        "Based on the above hierarchy, explain in clear, full sentences and in a conversational tone "
        "how these ideas might be causally connected. Emphasize unexpected insights, novel interconnections, "
        "and let the nodes catalyze each other like nodes of Ranvier. Your narrative should be creative, thoughtful, "
        "and expressed in complete sentences.\n\n"
        "=== Start Narrative ===\nNarrative:"
    )
    return prompt

# -------------------------------------------------------------------
# SENTIMENT & NOVELTY EVALUATION
# -------------------------------------------------------------------
sentiment_pipeline = pipeline("sentiment-analysis", device=0 if torch.cuda.is_available() else -1)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

def evaluate_sample(sample: str, reference_texts: list):
    is_coherent, coherence_score = calculate_coherency_score(sample, reference_texts)
    try:
        sent_result = sentiment_pipeline(sample)[0]
        sentiment_score = sent_result["score"] if sent_result["label"] == "POSITIVE" else 0
    except Exception as e:
        logger.exception("Sentiment analysis failed.")
        sentiment_score = 0
    effective_sentiment = sentiment_score + 0.1
    try:
        if reference_texts:
            ref_embs = embedding_model.encode(reference_texts, convert_to_tensor=True)
        else:
            ref_embs = embedding_model.encode([sample], convert_to_tensor=True)
        sample_emb = embedding_model.encode(sample, convert_to_tensor=True)
        sims = util.pytorch_cos_sim(sample_emb, ref_embs).cpu().numpy()[0]
        novelty_score = 1 - max(sims)
    except Exception as e:
        logger.exception("Novelty calculation failed.")
        novelty_score = 0
    combined_score = coherence_score * novelty_score * effective_sentiment
    return combined_score, coherence_score, novelty_score, sentiment_score

# -------------------------------------------------------------------
# TEXT GENERATION: DYNAMIC NARRATIVE GENERATION
# -------------------------------------------------------------------
text_generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M", device=LEXICAL_DEVICE)

def generate_text_from_prompt(prompt: str, extra_tokens=300) -> str:
    max_positions = 512
    tokens = prompt.split()
    if len(tokens) >= max_positions:
        prompt = " ".join(tokens[-(max_positions - 1):])
    effective_extra = min(extra_tokens, max_positions - len(prompt.split()) - 1)
    if effective_extra <= 0:
        effective_extra = 50
    tokenizer = text_generator.tokenizer
    model = text_generator.model
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(LEXICAL_DEVICE) for k, v in inputs.items()}
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=effective_extra,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    result = gen_text[len(prompt):].strip()
    result = remove_repetitions(result)
    result = truncate_text(result, max_words=MAX_GEN_WORDS)
    if len(result.split()) >= 20:
        return result
    return result

def refine_narrative(initial_narrative: str, base_prompt: str) -> str:
    refinement_prompt = (
        base_prompt + "\n\nHere is an initial narrative:\n" + initial_narrative +
        "\n\nNow, refine this narrative so that it is expressed in clear, coherent full sentences. "
        "Explain in detail how these topics interconnect like catalysts bridging nodes of Ranvier. "
        "Encourage creativity and logical coherence in your reasoning. Write your refined narrative in a conversational tone.\n\nRefined Narrative:"
    )
    refined_text = generate_text_from_prompt(refinement_prompt, extra_tokens=300)
    refined_text = remove_repetitions(refined_text)
    refined_text = truncate_text(refined_text, max_words=MAX_GEN_WORDS)
    return refined_text

def append_statistical_connections(narrative: str, hierarchy: dict) -> str:
    nodes = set()
    for parent, children in hierarchy.items():
        nodes.add(parent)
        nodes.update(children)
    if not nodes:
        return narrative
    nodes = list(nodes)
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    narrative_emb = embedder.encode(narrative, convert_to_tensor=True)
    node_embs = embedder.encode(nodes, convert_to_tensor=True)
    sims = util.pytorch_cos_sim(narrative_emb, node_embs).cpu().numpy()[0]
    top_indices = np.argsort(sims)[-3:][::-1]
    top_nodes = [nodes[i] for i in top_indices]
    connection_sentence = "Statistically, the concepts " + ", ".join(top_nodes) + " are strongly interrelated, reinforcing the overall logical coherence."
    return narrative + "\n\n" + connection_sentence

def generate_dynamic_conclusion(root: str, hierarchy: dict, summaries: dict) -> str:
    new_seed = random.randint(0, 1_000_000)
    set_seed(new_seed)
    logger.info(f"Random seed set to: {new_seed}")
    neurotransmitter_level = random.uniform(0.8, 1.2)
    logger.info(f"Neurotransmitter level set to: {neurotransmitter_level:.2f}")
    prompt = generate_initial_prompt(root, hierarchy, summaries)
    logger.info(f"Generated prompt for narrative:\n{prompt}")
    initial_narrative = generate_text_from_prompt(prompt, extra_tokens=300)
    logger.info(f"Initial Narrative:\n{initial_narrative}")
    combined_initial, c_initial, n_initial, s_initial = evaluate_sample(initial_narrative, list(summaries.values()))
    logger.info(f"Initial evaluation -- Combined: {combined_initial:.3f}, Coherence: {c_initial:.3f}, Novelty: {n_initial:.3f}, Sentiment: {s_initial:.3f}")
    refined_narrative = refine_narrative(initial_narrative, prompt)
    logger.info(f"Refined Narrative:\n{refined_narrative}")
    combined_refined, c_refined, n_refined, s_refined = evaluate_sample(refined_narrative, list(summaries.values()))
    logger.info(f"Refined evaluation -- Combined: {combined_refined:.3f}, Coherence: {c_refined:.3f}, Novelty: {n_refined:.3f}, Sentiment: {s_refined:.3f}")
    effective_sent_initial = (s_initial if s_initial > 0 else 0.1) * neurotransmitter_level
    adjusted_initial = c_initial * n_initial * effective_sent_initial
    effective_sent_refined = (s_refined if s_refined > 0 else 0.1) * neurotransmitter_level
    adjusted_refined = c_refined * n_refined * effective_sent_refined
    logger.info(f"Adjusted evaluation -- Initial: {adjusted_initial:.3f}, Refined: {adjusted_refined:.3f}")
    # Set minimum threshold to 0.13 instead of 0.1.
    MIN_THRESHOLD = 0.13
    if adjusted_refined >= MIN_THRESHOLD:
        final_narrative = refined_narrative if adjusted_refined >= adjusted_initial else initial_narrative
    elif adjusted_initial >= MIN_THRESHOLD:
        final_narrative = initial_narrative
    else:
        logger.info("No sample met the minimum threshold; using refined narrative anyway.")
        final_narrative = refined_narrative
    logger.info(f"Chosen narrative (adjusted combined score: {max(adjusted_initial, adjusted_refined):.3f}):\n{final_narrative}")
    final_output = remove_repetitions(prompt + "\n" + final_narrative)
    final_output = truncate_text(final_output, max_words=MAX_GEN_WORDS)
    return final_output

# -------------------------------------------------------------------
# COLLECT TRAINING SAMPLES
# -------------------------------------------------------------------
def collect_training_samples(num_samples=20, combined_threshold=0.2, max_iterations=50) -> str:
    selected_samples = []
    best_score = 0.0
    best_sample = None
    iteration = 0
    while len(selected_samples) < num_samples and iteration < max_iterations:
        reset_random_seed()
        iteration += 1
        logger.info(f"\n=== Sample Iteration {iteration} ===")
        summaries = get_wikipedia_data(num_articles=10)
        if not summaries:
            logger.error("No valid Wikipedia data found. Skipping iteration.")
            continue
        connections = encode_and_find_connections(summaries, threshold=0.3)
        if not connections:
            logger.error("No meaningful connections found. Skipping iteration.")
            continue
        graph = build_graph(connections)
        root, hierarchy = build_concept_hierarchy(graph)
        if not root:
            logger.error("Failed to build a concept hierarchy. Skipping iteration.")
            continue
        emergent_conclusion = generate_dynamic_conclusion(root, hierarchy, summaries)
        if not emergent_conclusion:
            logger.error("Failed to generate a valid conclusion. Skipping iteration.")
            continue
        combined_score, c_score, n_score, s_score = evaluate_sample(emergent_conclusion, list(summaries.values()))
        logger.info(f"Evaluation -- Combined: {combined_score:.3f}, Coherence: {c_score:.3f}, Novelty: {n_score:.3f}, Sentiment: {s_score:.3f}")
        # Use the new threshold of 0.13 for accepting samples.
        if combined_score >= 0.13:
            selected_samples.append(emergent_conclusion)
            logger.info(f"Accepted sample with score: {combined_score:.3f}. Total accepted: {len(selected_samples)}")
            if combined_score > best_score:
                best_score = combined_score
                best_sample = emergent_conclusion
                logger.info(f"New best sample updated with score: {best_score:.3f}")
        else:
            logger.info(f"Rejected sample; combined score {combined_score:.3f} is below the minimum threshold (0.13).")
        update_homeostatic_state(emergent_conclusion)
    if not selected_samples and best_sample is not None:
        logger.warning("No sample beat the best score threshold; using best sample anyway.")
        selected_samples.append(best_sample)
    if iteration >= max_iterations and len(selected_samples) < num_samples:
        logger.warning("Max iterations reached; returning collected samples.")
    return "\n\n==== NEW SAMPLE ====\n\n".join(selected_samples)

# -------------------------------------------------------------------
# TRAIN MODEL FUNCTION (For each branch separately)
# -------------------------------------------------------------------
def train_model_from_scratch(training_data: str, output_dir: str, config: dict,
                             epochs=3, device_for_training=torch.device("cpu"), batch_size=1):
    from transformers import GPT2LMHeadModel, AutoTokenizer, GPT2Config
    config_obj = GPT2Config(**config)
    if os.path.exists(output_dir) and os.path.isdir(output_dir) and any(
        os.path.exists(os.path.join(output_dir, f)) for f in ["config.json", "model.safetensors", "pytorch_model.bin"]
    ):
        logger.info(f"Model found in '{output_dir}'. Loading existing model (local_files_only=True)...")
        model = GPT2LMHeadModel.from_pretrained(output_dir, local_files_only=True).to(device_for_training)
        tokenizer = AutoTokenizer.from_pretrained(output_dir, local_files_only=True)
        logger.info(f"Existing model loaded from '{output_dir}'.")
    else:
        logger.info(f"No model found in '{output_dir}'. Initializing a new model from scratch.")
        model = GPT2LMHeadModel(config_obj).to(device_for_training)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        logger.info(f"New model initialized for '{output_dir}'.")
    # Remove the output directory if it exists to avoid file locks.
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    # Save using legacy serialization.
    model.save_pretrained(output_dir, safe_serialization=False)
    tokenizer.save_pretrained(output_dir)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tmpfile:
        tmpfile_path = tmpfile.name
        tmpfile.write(training_data)
    train_dataset = TextDataset(tokenizer=tokenizer, file_path=tmpfile_path, block_size=256)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=10,  # Save every 10 steps
        save_total_limit=SAVE_TOTAL_LIMIT,
        logging_steps=25,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset
    )
    logger.info(f"🚀 Starting training for '{output_dir}' on device {device_for_training} with batch size {batch_size}...")
    trainer.train()
    safe_save_model(trainer, output_dir)
    tokenizer.save_pretrained(output_dir)
    os.remove(tmpfile_path)
    logger.info(f"✅ Training complete. Model and tokenizer saved to '{output_dir}'")

# -------------------------------------------------------------------
# MAIN FUNCTION: DUAL BRANCH (LEXICAL & STATISTICAL) TRAINING CONCURRENTLY
# -------------------------------------------------------------------
def main():
    if os.path.isdir("Lexical_Model") and os.path.exists(os.path.join("Lexical_Model", "config.json")):
        logger.info("Lexical branch model found in 'Lexical_Model'. It will be loaded and fine-tuned.")
    else:
        logger.info("Lexical branch model not found in 'Lexical_Model'. A new model will be initialized.")
    if os.path.isdir("Statistical_Model") and os.path.exists(os.path.join("Statistical_Model", "config.json")):
        logger.info("Statistical branch model found in 'Statistical_Model'. It will be loaded and fine-tuned.")
    else:
        logger.info("Statistical branch model not found in 'Statistical_Model'. A new model will be initialized.")
    try:
        reset_random_seed()
        logger.info("📚 Collecting emergent training samples...")
        training_text = collect_training_samples(num_samples=20, combined_threshold=0.2, max_iterations=50)
        logger.info("\n=== Collected Training Data ===\n")
        logger.info(training_text)
        lexical_training_text = f"LEXICAL CONTEXT:\n{homeostatic_state['conscious']['context']}\n\n{training_text}"
        statistical_training_text = f"STATISTICAL CONTEXT:\n{homeostatic_state['subconscious']['context']}\n\n{training_text}"
        lexical_thread = threading.Thread(
            target=train_model_from_scratch,
            args=(lexical_training_text, "Lexical_Model", LexicalConfig().to_dict()),
            kwargs={"epochs": 3, "device_for_training": LEXICAL_DEVICE, "batch_size": 4}
        )
        statistical_thread = threading.Thread(
            target=train_model_from_scratch,
            args=(statistical_training_text, "Statistical_Model", StatisticalConfig().to_dict()),
            kwargs={"epochs": 3, "device_for_training": STATISTICAL_DEVICE, "batch_size": 8}
        )
        logger.info("Starting concurrent training for both branches (Lexical & Statistical)...")
        lexical_thread.start()
        statistical_thread.start()
        lexical_thread.join()
        statistical_thread.join()
        logger.info("✅ Both branches have finished training.")
    except Exception as e:
        logger.exception("An error occurred during the main process.")

if __name__ == "__main__":
    main()

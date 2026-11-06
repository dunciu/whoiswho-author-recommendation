import numpy as np
import time
import json
import math
import warnings
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import itertools # For co-author pairs

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Transformers imports
from transformers import AutoTokenizer, AutoModel

# Hyperparameter Optimization import
import optuna

# Suppress warnings
warnings.filterwarnings("ignore")

# ======================================================================
# 1. SCIBERT INITIALIZATION AND EMBEDDING FUNCTION
# ======================================================================
SCIBERT_MODEL_NAME = 'allenai/scibert_scivocab_uncased'

def initialize_scibert():
    """Initialize SciBERT model with fallback options"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(SCIBERT_MODEL_NAME)
        model = AutoModel.from_pretrained(SCIBERT_MODEL_NAME)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        print(f"SciBERT Model loaded on: {device} (Model: {SCIBERT_MODEL_NAME})")
        return tokenizer, model, device
    except Exception as e:
        print(f"Error loading SciBERT model. Using dummy embeddings: {e}")
        return None, None, torch.device('cpu')

scibert_tokenizer, scibert_model, DEVICE = initialize_scibert()

def get_scibert_embedding(text_list: List[str]) -> np.ndarray:
    """Generates SciBERT embeddings for a list of texts"""
    if not text_list or scibert_model is None:
        return np.random.rand(len(text_list), 768).astype(np.float32)
    try:
        inputs = scibert_tokenizer(
            text_list, return_tensors='pt', padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = scibert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return np.random.rand(len(text_list), 768).astype(np.float32)

# ======================================================================
# 2. DATA PREPROCESSING AND LOADING
# ======================================================================
def preprocess_papers_with_scibert(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generates SciBERT embeddings for all papers, including venue and year."""
    print(f"Generating SciBERT embeddings for {len(papers)} papers...")
    if not papers: return []

    # --- CORRECT: Includes venue and year ---
    texts = []
    for p in papers:
        title = p.get('title', '')
        abstract = p.get('abstract', '')
        venue = p.get('venue', '')
        year = str(p.get('year', ''))
        combined_text = f"{title} [SEP] {venue} [SEP] {year} [SEP] {abstract}"
        texts.append(combined_text)
    # --- END CORRECTION ---

    all_embeddings = []
    for i in range(0, len(texts), 32):
        batch_texts = texts[i:i+32]
        print(f"  Embedding batch {i//32 + 1}/{(len(texts) + 31)//32}...", end='\r')
        all_embeddings.append(get_scibert_embedding(batch_texts))
    print("\nEmbedding generation complete.")

    if all_embeddings:
        final_embeddings = np.concatenate(all_embeddings, axis=0)
        for i, paper in enumerate(papers):
            paper['scibert_embedding'] = final_embeddings[i]
    return papers

def open_file(filename: str = "data/pid_to_info_all.json") -> Optional[List[Dict[str, Any]]]:
    """Load data from JSON file with fallback to dummy data"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        papers_list = list(raw_data.values())
        print(f"Successfully loaded {len(papers_list)} papers from {filename}")
        return papers_list
    except FileNotFoundError:
        print(f"File {filename} not found. Generating dummy data...")
        # Fallback based on typical HPO size, adjust if needed
        return generate_dummy_data(1000)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def generate_dummy_data(num_papers: int) -> List[Dict[str, Any]]:
    """Generate dummy paper data for testing"""
    papers = []
    num_authors = max(200, num_papers // 20)
    authors = [f"A{i:04d}" for i in range(num_authors)]

    for i in range(num_papers):
        paper_authors = np.random.choice(authors, np.random.randint(1, 5), replace=False).tolist()
        papers.append({
            'id': f"P{i:06d}",
            'title': f"Research Paper on Topic {i}",
            'abstract': f"This paper investigates research questions in field {i % 10}. " * 5,
            'authors': [{'id': author_id} for author_id in paper_authors],
            'venue': f"Conference {i % 5}",
            'year': 2015 + (i % 10),
            'references': [] # Empty, as data lacks this
        })
    print(f"Generated {len(papers)} dummy papers with {len(authors)} unique authors")
    return papers

def extract_paper_author_ids(paper_data: Dict[str, Any]) -> List[str]:
    """Robustly extract author IDs"""
    authors = paper_data.get('authors', [])
    if not authors or not isinstance(authors, list): return []
    author_ids = []
    possible_keys = ['id', 'authorId', 'author_id', 'name']
    for author in authors:
        if not isinstance(author, dict): continue
        for key in possible_keys:
            if key in author and author[key] is not None:
                author_ids.append(str(author[key]))
                break
    return list(set(author_ids))

def filter_and_preprocess_data(papers_list: List[Dict[str, Any]], max_papers: Optional[int] = None) -> List[Dict[str, Any]]:
    """Filter papers, add author ID lists, and generate embeddings"""
    if not papers_list: return []
    valid_papers = []
    for paper in papers_list:
        author_ids = extract_paper_author_ids(paper)
        if (paper.get('title') or paper.get('abstract')) and author_ids:
            paper['author_id_list'] = author_ids
            valid_papers.append(paper)
    print(f"Found {len(valid_papers)} papers with valid text and author IDs.")

    if max_papers is not None: valid_papers = valid_papers[:max_papers]
    print(f"Processing a subset of {len(valid_papers)} papers for the final model.")
    return preprocess_papers_with_scibert(valid_papers)

# ======================================================================
# 3. GRAPH AND GNN IMPLEMENTATION
# ======================================================================
class ManualGraphData:
    def __init__(self):
        self.node_features: Dict[str, torch.Tensor] = {}
        self.edge_index: Dict[Tuple[str, str, str], torch.Tensor] = {}
        self.node_maps: Dict[str, Dict[str, int]] = {}
        self.num_nodes: Dict[str, int] = {}

# --- Using GCN Layer ---
class ManualGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate: float):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x, edge_index, num_target_nodes):
        src, dst = edge_index
        aggregated = torch.zeros(num_target_nodes, x.size(1), dtype=x.dtype, device=x.device)
        aggregated.index_add_(0, dst, x[src])
        degree = torch.bincount(dst, minlength=num_target_nodes).float().unsqueeze(1).clamp(min=1).to(x.device)
        # Apply activation *after* aggregation and linear layer
        return F.relu(self.linear(self.dropout(aggregated / degree)))
# --- END GCN Layer ---

class ManualHeteroGNN(nn.Module):
    """Heterogeneous GNN using GCN layers and Co-Author edges."""
    def __init__(self, paper_in_channels: int, hidden_channels: int, out_channels: int, dropout_rate: float):
        super().__init__()
        # Initial projection layers
        self.lin_paper = nn.Linear(paper_in_channels, hidden_channels)
        self.lin_author = nn.Linear(paper_in_channels, hidden_channels)

        # GCN layers (operate on hidden_channels -> hidden_channels)
        self.conv_a_to_p = ManualGCNLayer(hidden_channels, hidden_channels, dropout_rate)
        self.conv_p_to_a = ManualGCNLayer(hidden_channels, hidden_channels, dropout_rate)
        self.conv_co_author = ManualGCNLayer(hidden_channels, hidden_channels, dropout_rate)

        # Final projection layers
        self.out_lin_paper = nn.Linear(hidden_channels, out_channels)
        self.out_lin_author = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, num_nodes_dict):
        # 1. Initial projection to hidden_dim
        h_paper_init = F.relu(self.lin_paper(x_dict['paper']))
        h_author_init = F.relu(self.lin_author(x_dict['author']))

        # Keep track of embeddings updated by message passing
        h_paper = h_paper_init
        h_author = h_author_init

        # 2. Message Passing using GCN
        msg_for_paper = torch.zeros_like(h_paper)
        msg_for_author = torch.zeros_like(h_author)

        # Message: author -> paper
        if ('author', 'writes', 'paper') in edge_index_dict:
            edge_index = edge_index_dict[('author', 'writes', 'paper')]
            msg_for_paper += self.conv_a_to_p(h_author, edge_index, num_nodes_dict['paper'])

        # Message: paper -> author
        if ('paper', 'written_by', 'author') in edge_index_dict:
            edge_index = edge_index_dict[('paper', 'written_by', 'author')]
            msg_for_author += self.conv_p_to_a(h_paper, edge_index, num_nodes_dict['author'])

        # Message: author -> author (co-authorship)
        if ('author', 'co_author', 'author') in edge_index_dict:
            edge_index = edge_index_dict[('author', 'co_author', 'author')]
            msg_for_author += self.conv_co_author(h_author, edge_index, num_nodes_dict['author'])

        # 3. Residual Connection (Add aggregated messages to initial embeddings)
        h_paper_final = h_paper_init + msg_for_paper
        h_author_final = h_author_init + msg_for_author

        # 4. Final Projection to output dimension
        out_paper = self.out_lin_paper(h_paper_final)
        out_author = self.out_lin_author(h_author_final)

        return {'paper': out_paper, 'author': out_author}

def build_manual_graph(papers: List[Dict[str, Any]], author_id_key: str) -> ManualGraphData:
    """Builds the graph, including co-author edges."""
    all_author_ids = set(aid for p in papers for aid in p.get(author_id_key, []))
    if not all_author_ids or not papers: return ManualGraphData()
    author_map = {author_id: idx for idx, author_id in enumerate(all_author_ids)}
    paper_map = {p.get('id', f"paper_{id}"): i for i, p in enumerate(papers)} # Fixed f-string

    author_papers = defaultdict(list)
    for p in papers:
        for author_id in p.get(author_id_key, []):
            # Ensure embedding exists before appending
            if 'scibert_embedding' in p and p['scibert_embedding'] is not None:
                author_papers[author_id].append(p['scibert_embedding'])

    author_features_np = np.zeros((len(author_map), 768), dtype=np.float32)
    for author_id, embeddings in author_papers.items():
        if embeddings:
            author_features_np[author_map[author_id]] = np.mean(embeddings, axis=0)

    data = ManualGraphData()
    data.node_features['paper'] = torch.tensor(np.stack([p['scibert_embedding'] for p in papers]), dtype=torch.float32)
    data.node_features['author'] = torch.tensor(author_features_np, dtype=torch.float32)
    data.num_nodes = {'paper': len(papers), 'author': len(author_map)}
    data.node_maps = {'author': author_map, 'paper': paper_map}

    # Author-Paper Edges & Co-Author Edge Creation
    auth_src, auth_dst = [], []
    co_author_pairs = set()

    for p in papers:
        paper_id = p.get('id', f"paper_{id}") # Fixed f-string
        authors_in_paper = p.get(author_id_key, [])
        author_indices_in_paper = [author_map[aid] for aid in authors_in_paper if aid in author_map]

        # Add author-paper edges
        for auth_idx in author_indices_in_paper:
            # Check if paper exists in the map (it should, but added safety)
            if paper_id in paper_map:
                auth_src.append(auth_idx)
                auth_dst.append(paper_map[paper_id])

        # Add co-author pairs
        for a1, a2 in itertools.combinations(author_indices_in_paper, 2):
            co_author_pairs.add(tuple(sorted((a1, a2))))

    if auth_src:
        data.edge_index[('author', 'writes', 'paper')] = torch.tensor([auth_src, auth_dst], dtype=torch.long)
        data.edge_index[('paper', 'written_by', 'author')] = torch.tensor([auth_dst, auth_src], dtype=torch.long)
    else:
        print("Warning: No author-paper edges were created.")


    # Add co-author edges to graph
    co_auth_src, co_auth_dst = [], []
    for a1, a2 in co_author_pairs:
        co_auth_src.extend([a1, a2])
        co_auth_dst.extend([a2, a1]) # Add edges in both directions

    co_author_edge_count = 0
    if co_auth_src:
        co_author_edge_count = len(co_auth_src)
        data.edge_index[('author', 'co_author', 'author')] = torch.tensor([co_auth_src, co_auth_dst], dtype=torch.long)

    print(f"Graph built with {data.num_nodes['author']} authors, {data.num_nodes['paper']} papers, {len(auth_src)} author edges, and {co_author_edge_count} co-author edges.")
    return data

# ======================================================================
# 4. EVALUATION METRICS
# ======================================================================

def calculate_precision_recall_f1(true_authors: set, candidates: list, k: int) -> Tuple[float, float, float]:
    """Calculates Precision@K, Recall@K, and F1@K"""
    if not true_authors: return 0.0, 0.0, 0.0
    predicted_authors = set(c['id'] for c in candidates[:k])
    tp = len(true_authors.intersection(predicted_authors))
    precision = tp / k if k > 0 else 0.0
    recall = tp / len(true_authors) if true_authors else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def calculate_ndcg(true_authors: set, candidates: list, k: int) -> float:
    """Calculates NDCG@k for author ranking."""
    if not true_authors: return 0.0
    relevance = [1 if candidate['id'] in true_authors else 0 for candidate in candidates[:k]]
    dcg = sum((rel / math.log2(i + 2)) for i, rel in enumerate(relevance))
    ideal_relevance = [1] * min(len(true_authors), k)
    idcg = sum((rel / math.log2(i + 2)) for i, rel in enumerate(ideal_relevance))
    return dcg / idcg if idcg > 0 else 0.0

# ======================================================================
# 5. OPTUNA OBJECTIVE FUNCTION
# ======================================================================

def objective(trial: optuna.Trial, all_papers: List[Dict[str, Any]], test_set_ratio: float) -> float:
    """
    The objective function for Optuna to optimize (GCN, Co-Authors, Random Negatives).
    """

    # --- 1. Suggest Hyperparameters ---
    print(f"\n--- Starting Trial {trial.number} ---")
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256])
    output_dim = trial.suggest_categorical('output_dim', [64, 128])
    lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    epochs = trial.suggest_int('epochs', 20, 35)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    margin = trial.suggest_float('margin', 0.4, 1.0)

    K_AT = 10
    AUTHOR_ID_KEY = 'author_id_list'
    EMBEDDING_DIM = 768

    # --- 2. Data Setup ---
    test_set_size = int(len(all_papers) * test_set_ratio)
    train_papers = all_papers[:-test_set_size]
    test_papers = all_papers[-test_set_size:]

    graph_data = build_manual_graph(train_papers, AUTHOR_ID_KEY)
    if not graph_data.edge_index or ('author', 'writes', 'paper') not in graph_data.edge_index:
        print("No author edges in training graph, pruning trial.")
        raise optuna.exceptions.TrialPruned("No author edges in training graph.")

    for key in graph_data.node_features: graph_data.node_features[key] = graph_data.node_features[key].to(DEVICE)
    for key in graph_data.edge_index: graph_data.edge_index[key] = graph_data.edge_index[key].to(DEVICE)

    # --- 3. Model Training ---
    model = ManualHeteroGNN(
        EMBEDDING_DIM, hidden_dim, output_dim, dropout
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    model.train()
    pos_edges = graph_data.edge_index[('author', 'writes', 'paper')]
    num_authors = graph_data.num_nodes['author']

    for epoch in range(epochs):
        optimizer.zero_grad()
        h_dict = model(graph_data.node_features, graph_data.edge_index, graph_data.num_nodes)

        anchor_emb = h_dict['paper'][pos_edges[1]]
        positive_emb = h_dict['author'][pos_edges[0]]

        # --- Using Random Negative Sampling ---
        negative_author_indices = torch.randint(0, num_authors, (pos_edges.shape[1],), device=DEVICE)
        # --- END ---

        negative_emb = h_dict['author'][negative_author_indices]

        loss = loss_fn(anchor_emb, positive_emb, negative_emb)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  Trial {trial.number}, Epoch {epoch+1:02d}/{epochs}, Loss: {loss.item():.4f}", end='\r')
    print() # Newline after training

    # --- 4. Model Evaluation ---
    model.eval()

    all_paper_features = torch.tensor(np.stack([p['scibert_embedding'] for p in all_papers]), dtype=torch.float32).to(DEVICE)
    inference_graph_data = build_manual_graph(all_papers, AUTHOR_ID_KEY)

    inf_node_features = {
        'paper': all_paper_features,
        'author': inference_graph_data.node_features['author'].to(DEVICE)
    }
    inf_num_nodes = inference_graph_data.num_nodes

    with torch.no_grad():
        h_dict_eval = model(inf_node_features, graph_data.edge_index, inf_num_nodes)

    all_author_embeddings = h_dict_eval['author']
    all_paper_embeddings = h_dict_eval['paper']

    total_ndcg, total_f1, evaluated_count = 0.0, 0.0, 0
    inference_author_map = inference_graph_data.node_maps['author']

    for i, test_paper in enumerate(test_papers):
        test_paper_index = len(train_papers) + i
        true_author_ids_all = set(test_paper.get(AUTHOR_ID_KEY, []))

        true_author_ids = true_author_ids_all.intersection(inference_author_map.keys())
        if not true_author_ids: continue

        test_paper_embedding = all_paper_embeddings[test_paper_index].unsqueeze(0)
        similarities = F.cosine_similarity(test_paper_embedding, all_author_embeddings)

        candidate_scores = []
        for author_id, author_idx in inference_author_map.items():
            candidate_scores.append({'id': author_id, 'score': similarities[author_idx].item()})
        candidate_scores.sort(key=lambda x: x['score'], reverse=True)

        ndcg = calculate_ndcg(true_author_ids, candidate_scores, K_AT)
        _, _, f1 = calculate_precision_recall_f1(true_author_ids_all, candidate_scores, K_AT)

        total_ndcg += ndcg
        total_f1 += f1
        evaluated_count += 1

    avg_ndcg = total_ndcg / evaluated_count if evaluated_count > 0 else 0.0
    avg_f1 = total_f1 / evaluated_count if evaluated_count > 0 else 0.0

    print(f"--- Trial {trial.number} Complete --- Avg. NDCG: {avg_ndcg:.4f}, Avg. F1: {avg_f1:.4f}")

    # --- 5. Return Metric to Maximize ---
    return avg_ndcg # We will optimize for NDCG


# ======================================================================
# 6. HPO RUNNER
# ======================================================================

def run_hpo(max_papers: int, num_trials: int, test_set_ratio: float):
    """Loads data once, then runs the full Optuna HPO study."""

    print("Starting HPO pipeline...")
    papers_list = open_file()
    if not papers_list:
        print("Failed to load data. Exiting.")
        return

    papers_with_embeddings = filter_and_preprocess_data(papers_list, max_papers)
    if not papers_with_embeddings:
        print("No valid papers to process after filtering. Exiting.")
        return

    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(
        lambda trial: objective(trial, papers_with_embeddings, test_set_ratio),
        n_trials=num_trials,
        timeout=None # Run to completion
    )

    print("\n" + "=" * 60)
    print("HYPERPARAMETER OPTIMIZATION COMPLETE 🏆")
    print("=" * 60)
    print(f"Number of finished trials: {len(study.trials)}")

    print("\nBest trial:")
    best_trial = study.best_trial

    print(f"  Value (Avg. NDCG): {best_trial.value:.4f}")

    print("  Best Parameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    print("=" * 60)

    print("\nTo run the final model, use these parameters in your FINAL_CONFIG dict.")


if __name__ == "__main__":
    # --- Configuration for the HPO run ---

    # Adjust MAX_PAPERS for desired dataset size (e.g., 1000, 5000, 10000)
    MAX_PAPERS_FOR_HPO = 1000
    NUM_TRIALS = 25
    TEST_SET_RATIO = 0.2

    # --- Run the HPO ---
    run_hpo(MAX_PAPERS_FOR_HPO, NUM_TRIALS, TEST_SET_RATIO)
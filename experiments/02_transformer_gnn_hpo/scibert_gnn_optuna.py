import numpy as np
import time
import json
import math
import warnings
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

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
    """Generates SciBERT embeddings for all papers"""
    print(f"Generating SciBERT embeddings for {len(papers)} papers...")
    if not papers: return []
    texts = [f"{p.get('title', '')} [SEP] {p.get('abstract', '')}" for p in papers]
    
    all_embeddings = []
    for i in range(0, len(texts), 32):
        all_embeddings.append(get_scibert_embedding(texts[i:i + 32]))
    
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
        return generate_dummy_data(500)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def generate_dummy_data(num_papers: int) -> List[Dict[str, Any]]:
    """Generate dummy paper data for testing"""
    papers, authors = [], [f"A{i:03d}" for i in range(100)]
    for i in range(num_papers):
        paper_authors = np.random.choice(authors, np.random.randint(1, 5), replace=False).tolist()
        papers.append({
            'id': f"P{i:05d}", 'title': f"Research Paper on Topic {i}",
            'abstract': f"This paper investigates research questions in field {i % 10}. " * 5,
            'authors': [{'id': author_id} for author_id in paper_authors]
        })
    print(f"Generated {len(papers)} dummy papers with {len(authors)} unique authors")
    return papers

def extract_paper_author_ids(paper_data: Dict[str, Any]) -> List[str]:
    """
    FIX: Robustly extract author IDs by checking for multiple common key names.
    This is the critical fix for the "0 valid papers" issue.
    """
    authors = paper_data.get('authors', [])
    if not authors or not isinstance(authors, list):
        return []

    author_ids = []
    possible_keys = ['id', 'authorId', 'author_id', 'name'] # Add other potential keys here

    for author in authors:
        if not isinstance(author, dict):
            continue
        
        found_id = None
        for key in possible_keys:
            if key in author and author[key] is not None:
                found_id = author[key]
                break # Stop at the first valid key found
        
        if found_id:
            author_ids.append(str(found_id))
            
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

    if max_papers is not None:
        valid_papers = valid_papers[:max_papers]
    
    print(f"Processing {len(valid_papers)} valid papers for the experiment.")
    return preprocess_papers_with_scibert(valid_papers)

# ======================================================================
# ... THE REST OF THE SCRIPT (SECTIONS 3, 4, 5, 6) REMAINS UNCHANGED ...
# ======================================================================

class ManualGraphData:
    """Manual graph data structure"""
    def __init__(self):
        self.node_features: Dict[str, torch.Tensor] = {}
        self.edge_index: Dict[Tuple[str, str, str], torch.Tensor] = {}
        self.node_maps: Dict[str, Dict[str, int]] = {}
        self.num_nodes: Dict[str, int] = {}

class ManualGCNLayer(nn.Module):
    """Manual GCN layer with mean aggregation"""
    def __init__(self, in_channels, out_channels, dropout_rate: float):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, edge_index, num_target_nodes):
        src, dst = edge_index
        aggregated = torch.zeros(num_target_nodes, x.size(1), dtype=x.dtype, device=x.device)
        aggregated.index_add_(0, dst, x[src])
        degree = torch.bincount(dst, minlength=num_target_nodes).float().unsqueeze(1).clamp(min=1).to(x.device)
        return self.linear(self.dropout(aggregated / degree))

class ManualHeteroGNN(nn.Module):
    """Manual Heterogeneous GNN for learning embeddings"""
    def __init__(self, paper_in_channels: int, hidden_channels: int, out_channels: int, dropout_rate: float):
        super().__init__()
        self.lin_paper = nn.Linear(paper_in_channels, hidden_channels)
        self.lin_author = nn.Linear(paper_in_channels, hidden_channels)
        
        self.conv1 = ManualGCNLayer(hidden_channels, hidden_channels, dropout_rate)
        self.conv2 = ManualGCNLayer(hidden_channels, out_channels, dropout_rate)
        
    def forward(self, x_dict, edge_index_dict, num_nodes_dict):
        x_paper = F.relu(self.lin_paper(x_dict['paper']))
        x_author = F.relu(self.lin_author(x_dict['author']))
        
        if ('author', 'writes', 'paper') in edge_index_dict:
            edge_index = edge_index_dict[('author', 'writes', 'paper')]
            x_paper = x_paper + F.relu(self.conv1(x_author, edge_index, num_nodes_dict['paper']))
        
        if ('paper', 'written_by', 'author') in edge_index_dict:
            edge_index = edge_index_dict[('paper', 'written_by', 'author')]
            x_author = x_author + F.relu(self.conv1(x_paper, edge_index, num_nodes_dict['author']))
            
        if ('author', 'writes', 'paper') in edge_index_dict:
            edge_index = edge_index_dict[('author', 'writes', 'paper')]
            x_paper = x_paper + self.conv2(x_author, edge_index, num_nodes_dict['paper'])
            
        return {'paper': x_paper, 'author': x_author}

def build_manual_graph(papers: List[Dict[str, Any]], author_id_key: str) -> ManualGraphData:
    """Builds a heterogeneous graph with informative author features"""
    all_author_ids = set(aid for p in papers for aid in p.get(author_id_key, []))
    if not all_author_ids or not papers: return ManualGraphData()

    author_map = {author_id: idx for idx, author_id in enumerate(all_author_ids)}
    paper_map = {p.get('id', f"paper_{i}"): i for i, p in enumerate(papers)}

    author_papers = defaultdict(list)
    for i, p in enumerate(papers):
        for author_id in p.get(author_id_key, []):
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

    src, dst = [], []
    for i, p in enumerate(papers):
        paper_id = p.get('id', f"paper_{i}")
        for author_id in p.get(author_id_key, []):
            src.append(author_map[author_id])
            dst.append(paper_map[paper_id])
            
    if src:
        data.edge_index[('author', 'writes', 'paper')] = torch.tensor([src, dst], dtype=torch.long)
        data.edge_index[('paper', 'written_by', 'author')] = torch.tensor([dst, src], dtype=torch.long)
    print(f"Graph built with {data.num_nodes['author']} authors, {data.num_nodes['paper']} papers, and {len(src)} edges.")
    return data

def calculate_precision_recall_f1(true_authors: set, candidates: list, k: int) -> Tuple[float, float, float]:
    """Calculates Precision@K, Recall@K, and F1@K"""
    if not true_authors: return 0.0, 0.0, 0.0
    predicted_authors = set(c['id'] for c in candidates[:k])
    tp = len(true_authors.intersection(predicted_authors))
    precision = tp / k if k > 0 else 0.0
    recall = tp / len(true_authors) if true_authors else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def objective(trial: optuna.Trial, all_papers: List[Dict[str, Any]], test_set_ratio: float) -> float:
    """The objective function for Optuna to optimize using a ranking loss"""
    print(f"\n--- Starting Trial {trial.number} ---")
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256])
    output_dim = trial.suggest_categorical('output_dim', [64, 128])
    learning_rate = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    num_epochs = trial.suggest_int('epochs', 15, 30)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    dropout_rate = trial.suggest_float('dropout', 0.1, 0.4)
    margin = trial.suggest_float('margin', 0.2, 1.0)

    K_AT = 10
    AUTHOR_ID_KEY = 'author_id_list'
    EMBEDDING_DIM = 768

    test_set_size = int(len(all_papers) * test_set_ratio)
    train_papers = all_papers[:-test_set_size]
    test_papers = all_papers[-test_set_size:]

    graph_data = build_manual_graph(train_papers, AUTHOR_ID_KEY)
    if not graph_data.edge_index:
        raise optuna.exceptions.TrialPruned("No edges in the training graph.")

    for key in graph_data.node_features: graph_data.node_features[key] = graph_data.node_features[key].to(DEVICE)
    for key in graph_data.edge_index: graph_data.edge_index[key] = graph_data.edge_index[key].to(DEVICE)

    model = ManualHeteroGNN(EMBEDDING_DIM, hidden_dim, output_dim, dropout_rate).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    model.train()
    pos_edges = graph_data.edge_index[('author', 'writes', 'paper')]
    num_authors = graph_data.num_nodes['author']

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        h_dict = model(graph_data.node_features, graph_data.edge_index, graph_data.num_nodes)
        
        anchor_emb = h_dict['paper'][pos_edges[1]]
        positive_emb = h_dict['author'][pos_edges[0]]
        negative_author_indices = torch.randint(0, num_authors, (pos_edges.shape[1],), device=DEVICE)
        negative_emb = h_dict['author'][negative_author_indices]
        
        loss = loss_fn(anchor_emb, positive_emb, negative_emb)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Trial {trial.number}, Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    model.eval()
    all_paper_features = torch.tensor(np.stack([p['scibert_embedding'] for p in all_papers]), dtype=torch.float32).to(DEVICE)
    inference_graph_data = build_manual_graph(all_papers, AUTHOR_ID_KEY)
    inf_node_features = {
        'paper': all_paper_features,
        'author': inference_graph_data.node_features['author'].to(DEVICE)
    }

    with torch.no_grad():
        h_dict_eval = model(inf_node_features, graph_data.edge_index, inference_graph_data.num_nodes)
        all_author_embeddings = h_dict_eval['author']
        all_paper_embeddings = h_dict_eval['paper']
    
    total_f1 = 0.0
    evaluated_count = 0
    for i, test_paper in enumerate(test_papers):
        test_paper_index = len(train_papers) + i
        true_author_ids = set(test_paper.get(AUTHOR_ID_KEY, []))
        if not true_author_ids: continue

        test_paper_embedding = all_paper_embeddings[test_paper_index].unsqueeze(0)
        similarities = F.cosine_similarity(test_paper_embedding, all_author_embeddings)
        
        candidate_scores = []
        for author_id, author_idx in inference_graph_data.node_maps['author'].items():
            candidate_scores.append({'id': author_id, 'score': similarities[author_idx].item()})
        
        candidate_scores.sort(key=lambda x: x['score'], reverse=True)
        _, _, f1 = calculate_precision_recall_f1(true_author_ids, candidate_scores, K_AT)
        total_f1 += f1
        evaluated_count += 1
        
    avg_f1 = total_f1 / evaluated_count if evaluated_count > 0 else 0.0
    print(f"--- Trial {trial.number} Complete --- Avg. F1-Score: {avg_f1:.4f}")
    return avg_f1

def run_experiment_with_hpo():
    """Runs the full hyperparameter optimization experiment using Optuna."""
    MAX_PAPERS, TEST_SET_RATIO, NUM_TRIALS = 500, 0.25, 25

    print("Loading and preprocessing data for the entire experiment...")
    papers_list = open_file()
    if not papers_list:
        print("Failed to load data. Exiting."); return
    papers_with_embeddings = filter_and_preprocess_data(papers_list, MAX_PAPERS)

    # Check if we have enough data to proceed
    if not papers_with_embeddings:
        print("No valid papers found after filtering. Cannot start HPO. Exiting.")
        return

    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    try:
        study.optimize(
            lambda trial: objective(trial, papers_with_embeddings, TEST_SET_RATIO),
            n_trials=NUM_TRIALS, timeout=2700
        )
    except Exception as e:
        print(f"\nAn unexpected error occurred during the optimization: {e}")

    print("\n" + "=" * 60)
    print("HYPERPARAMETER OPTIMIZATION COMPLETE 🏆")
    print("=" * 60)

    try:
        best_trial = study.best_trial
        print(f"Number of finished trials: {len(study.trials)}")
        print(f"Best trial F1-Score: {best_trial.value:.4f}")
        print("Best Parameters:")
        for key, value in best_trial.params.items():
            print(f"  - {key}: {value}")
    except ValueError:
        print("The study finished, but NO trials were completed successfully.")
        print("This often means every trial failed or was pruned.")
        print("\n--- Trial States ---")
        failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        print(f"Total trials: {len(study.trials)}")
        print(f"Failed trials: {len(failed_trials)}")
        print(f"Pruned trials: {len(pruned_trials)}")
    print("=" * 60)

if __name__ == "__main__":
    run_experiment_with_hpo()
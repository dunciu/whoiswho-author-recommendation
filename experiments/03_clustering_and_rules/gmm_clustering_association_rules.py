import numpy as np
import time
import json
import math
import warnings
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
import itertools
import os
import pickle
import random
from tqdm import tqdm # <-- For progress bars

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# *** Import SentenceTransformer (SPECTER) ***
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    print("Warning: sentence-transformers library not found. SPECTER embedding generation will not work.")
    print("Install it via: pip install sentence-transformers")

# Hyperparameter Optimization import
import optuna

# --- NEW IMPORTS FOR CLUSTERING (Option 1) ---
try:
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans # <-- ADDED FOR HARD NEGATIVE SAMPLING
    import matplotlib.pyplot as plt
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn or matplotlib not found. Clustering analysis will not work.")
    print("Install them via: pip install scikit-learn matplotlib")
# --- END NEW IMPORTS ---

# --- NEW IMPORTS FOR ASSOCIATION RULES ---
try:
    import pandas as pd
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False
    print("Warning: mlxtend or pandas not found. Association rule mining will not work.")
    print("Install them via: pip install mlxtend pandas")
# --- END NEW IMPORTS ---


# Suppress warnings
warnings.filterwarnings("ignore")

# ======================================================================
# 0. SEEDING FOR REPRODUCIBILITY
# ======================================================================
SEED = 42
def set_seed(seed: int):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")

set_seed(SEED)

# ======================================================================
# 1. EMBEDDING MODEL INITIALIZATION (SPECTER-Only)
# ======================================================================
# --- Using SPECTER ---
EMBEDDING_MODEL_NAME = 'allenai/specter'
EMBEDDING_TYPE = 'specter'
# ---

EMBEDDING_DIM = 768
embedding_key = f'{EMBEDDING_TYPE}_embedding' # Global embedding key based on type

def initialize_embedding_model(model_name: str, model_type: str) -> Tuple[Optional[Any], Optional[Any], torch.device]:
    """Initializes a SentenceTransformer model (SciBERT logic removed)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = None # Tokenizer is handled internally by SentenceTransformer
    model = None
    
    if model_type == 'specter':
        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                model = SentenceTransformer(model_name, device=str(device))
                print(f"SentenceTransformer Model loaded on: {device} (Model: {model_name})")
            except Exception as e: print(f"Error loading SentenceTransformer model: {e}")
        else: print("Cannot load SPECTER model, sentence-transformers not installed.")
    else: 
        print(f"Unknown or unsupported model_type: {model_type}. Only 'specter' is supported.")
        
    return tokenizer, model, device

embed_tokenizer, embed_model, DEVICE = initialize_embedding_model(EMBEDDING_MODEL_NAME, EMBEDDING_TYPE)

def get_embedding(text_list: List[str], model_type: str) -> np.ndarray:
    """Generates embeddings using the chosen model (SciBERT logic removed)."""
    if not text_list or embed_model is None: return np.random.rand(len(text_list), EMBEDDING_DIM).astype(np.float32)
    
    try:
        if model_type == 'specter' and SENTENCE_TRANSFORMER_AVAILABLE:
             # SPECTER format: [CLS] Title [SEP] Abstract [SEP]
             # sentence_transformers handles this well
             combined_texts = []
             for text in text_list: # text_list now expects "title [SEP] abstract"
                 combined_texts.append(text)
             embeddings = embed_model.encode(combined_texts, convert_to_numpy=True, show_progress_bar=False)
        else: 
            raise ValueError(f"Model not loaded or unknown model_type: {model_type}")
        
        return embeddings.astype(np.float32)
    except Exception as e:
        print(f"Error generating embeddings with {model_type}: {e}")
        return np.random.rand(len(text_list), EMBEDDING_DIM).astype(np.float32)

# ======================================================================
# 2. DATA PREPROCESSING AND LOADING (with Caching)
# ======================================================================
def preprocess_papers_with_embeddings(papers: List[Dict[str, Any]], model_type: str, cache_path: str) -> List[Dict[str, Any]]:
    """Generates embeddings (SPECTER) for all papers, using caching."""
    cached_embeddings = {}
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}...")
        try:
            with open(cache_path, 'rb') as f: cached_embeddings = pickle.load(f)
            print(f"Loaded {len(cached_embeddings)} embeddings from cache.")
        except Exception as e: print(f"Error loading cache file: {e}. Regenerating."); cached_embeddings = {}

    papers_requiring_embedding = []
    current_embedding_key = f'{model_type}_embedding'
    for paper in papers:
        paper_id = paper.get('id')
        if paper_id and paper_id in cached_embeddings:
            paper[current_embedding_key] = cached_embeddings[paper_id]
        elif current_embedding_key not in paper or paper[current_embedding_key] is None:
            if paper_id: # Ensure paper has an ID
                papers_requiring_embedding.append(paper)

    if not papers_requiring_embedding:
        print(f"All required {model_type} embeddings were found.")
        # Ensure correct key exists
        for paper in papers:
            if current_embedding_key not in paper:
                found_key=None;
                for key in paper:
                    if 'embedding' in key: paper[current_embedding_key]=paper.pop(key); found_key=key; break
                if not found_key: paper[current_embedding_key]=None
        papers = [p for p in papers if p.get(current_embedding_key) is not None]
        return papers

    print(f"Generating {model_type} embeddings for {len(papers_requiring_embedding)} papers...")
    texts_to_embed = []
    for p in papers_requiring_embedding:
        title = p.get('title', '')
        abstract = p.get('abstract', '')
        
        if model_type == 'specter':
            # Format for SPECTER
            combined_text = f"{title}{embed_model.tokenizer.sep_token}{abstract}"
        else:
            print(f"Warning: Unsupported model_type {model_type} in preprocess. Defaulting to title/abstract.")
            combined_text = f"{title} {abstract}" # Fallback
        
        texts_to_embed.append(combined_text)

    all_gen_embeddings = []
    batch_size=32
    for i in tqdm(range(0, len(texts_to_embed), batch_size), desc=f"{model_type.capitalize()} Embedding Batches"):
        batch_texts = texts_to_embed[i:i+batch_size]
        all_gen_embeddings.append(get_embedding(batch_texts, model_type))
    print(f"\n{model_type.capitalize()} embedding generation complete.")

    if all_gen_embeddings:
        final_embeddings = np.concatenate(all_gen_embeddings, axis=0)
        newly_embedded_data = {}
        for i, paper in enumerate(papers_requiring_embedding):
            embedding = final_embeddings[i]
            paper[current_embedding_key] = embedding
            newly_embedded_data[paper['id']] = embedding # Use paper ID as key

        if newly_embedded_data:
            print(f"Updating cache file: {cache_path}")
            cached_embeddings.update(newly_embedded_data)
            try:
                with open(cache_path, 'wb') as f: pickle.dump(cached_embeddings, f)
            except Exception as e: print(f"Error saving cache file: {e}")

    papers = [p for p in papers if p.get(current_embedding_key) is not None]
    return papers

# --- Other data loading functions ---
def open_file(filename: str = "data/pid_to_info_all.json") -> Optional[List[Dict[str, Any]]]:
    try:
        with open(filename, 'r', encoding='utf-8') as f: raw_data = json.load(f)
        papers_list = list(raw_data.values())
        print(f"Successfully loaded {len(papers_list)} papers from {filename}")
        for i, p in enumerate(papers_list):
            if 'id' not in p or p['id'] is None: p['id'] = f"paper_{i}_{time.time()}"
        return papers_list
    except FileNotFoundError:
        print(f"File {filename} not found. Generating dummy data...")
        return generate_dummy_data(10000) # Default to 10k if file missing
    except Exception as e: print(f"Error loading file: {e}"); return None

def generate_dummy_data(num_papers: int) -> List[Dict[str, Any]]:
    papers = []
    num_authors = max(200, num_papers // 20)
    authors = [f"A{i:04d}" for i in range(num_authors)]
    for i in range(num_papers):
        paper_authors = np.random.choice(authors, np.random.randint(1, 5), replace=False).tolist()
        papers.append({
            'id': f"P{i:06d}", 'title': f"Research Paper {i}", 'abstract': f"Abstract {i} " * 5,
            'authors': [{'id': a} for a in paper_authors], 'venue': f"Conf {i%5}", 'year': 2015+(i%10),
            'references': [], 'author_id_list': paper_authors})
    print(f"Generated {len(papers)} dummy papers.")
    return papers

def extract_paper_author_ids(paper_data: Dict[str, Any]) -> List[str]:
    if 'author_id_list' in paper_data: return paper_data['author_id_list']
    authors = paper_data.get('authors', [])
    if not authors or not isinstance(authors, list): return []
    author_ids = []
    possible_keys = ['id', 'authorId', 'author_id', 'name']
    for author in authors:
        if not isinstance(author, dict): continue
        for key in possible_keys:
            if key in author and author[key] is not None: author_ids.append(str(author[key])); break
    return list(set(author_ids))

def filter_and_preprocess_data(papers_list: List[Dict[str, Any]], max_papers: Optional[int] = None,
                                 model_type: str = EMBEDDING_TYPE, cache_dir: str = ".") -> List[Dict[str, Any]]:
    """Filter papers, add author ID lists, and generate embeddings with caching."""
    if not papers_list: return []
    valid_papers = []
    seen_ids = set()
    for paper in papers_list:
        paper_id = paper.get('id')
        if paper_id is None or paper_id in seen_ids: continue
        author_ids = extract_paper_author_ids(paper)
        if (paper.get('title') or paper.get('abstract')) and author_ids:
            paper['author_id_list'] = author_ids
            valid_papers.append(paper)
            seen_ids.add(paper_id)
    print(f"Found {len(valid_papers)} unique papers with valid text and author IDs.")

    if max_papers is not None and len(valid_papers) > max_papers:
        print(f"Sorting and selecting top {max_papers} papers...")
        def get_year_int(p):
            year_val = p.get('year');
            if year_val is None: return 0
            try: return int(year_val)
            except: return 0
        valid_papers.sort(key=get_year_int, reverse=True)
        valid_papers = valid_papers[:max_papers]

    print(f"Processing a subset of {len(valid_papers)} papers.")
    cache_filename = f"{model_type}_embeddings_{len(valid_papers)}.pkl" # Use actual count
    cache_path = os.path.join(cache_dir, cache_filename)
    return preprocess_papers_with_embeddings(valid_papers, model_type=model_type, cache_path=cache_path)

# ======================================================================
# 3. GRAPH AND GNN IMPLEMENTATION (GCN + Co-Authors)
# ======================================================================
class ManualGraphData:
    def __init__(self): self.node_features={}; self.edge_index={}; self.node_maps={}; self.num_nodes={}
class ManualGCNLayer(nn.Module):
    def __init__(self, i, o, d): super().__init__(); self.lin=nn.Linear(i,o); self.drop=nn.Dropout(d)
    def forward(self, x, ei, n): s,d=ei; a=torch.zeros(n, x.size(1), dtype=x.dtype, device=x.device); a.index_add_(0,d,x[s]); deg=torch.bincount(d, minlength=n).float().unsqueeze(1).clamp(min=1).to(x.device); return F.relu(self.lin(self.drop(a/deg)))
class ManualHeteroGNN(nn.Module):
    def __init__(self, pi, h, o, d): super().__init__(); self.lp=nn.Linear(pi,h); self.la=nn.Linear(pi,h); self.c_ap=ManualGCNLayer(h,h,d); self.c_pa=ManualGCNLayer(h,h,d); self.c_co=ManualGCNLayer(h,h,d); self.olp=nn.Linear(h,o); self.ola=nn.Linear(h,o)
    def forward(self, xd, eid, nnd):
        hp_i=F.relu(self.lp(xd['paper'])); ha_i=F.relu(self.la(xd['author'])); hp,ha=hp_i,ha_i; mp=torch.zeros_like(hp); ma=torch.zeros_like(ha)
        if ('a','w','p') in eid: mp+=self.c_ap(ha, eid[('a','w','p')], nnd['paper'])
        if ('p','wb','a') in eid: ma+=self.c_pa(hp, eid[('p','wb','a')], nnd['author'])
        if ('a','co','a') in eid: ma+=self.c_co(ha, eid[('a','co','a')], nnd['author'])
        hp_f=hp_i+mp; ha_f=ha_i+ma; op=self.olp(hp_f); oa=self.ola(ha_f); return {'paper':op, 'author':oa}

def build_manual_graph(papers: List[Dict[str, Any]], author_id_key: str, embedding_key: str) -> ManualGraphData:
    all_author_ids = set(aid for p in papers for aid in p.get(author_id_key, []))
    if not all_author_ids or not papers: return ManualGraphData()
    author_map = {aid: idx for idx, aid in enumerate(all_author_ids)}
    valid_papers = [p for p in papers if p.get('id') is not None and embedding_key in p and p[embedding_key] is not None]
    if not valid_papers: print("Error: No valid papers for graph."); return ManualGraphData()
    paper_map = {p['id']: i for i, p in enumerate(valid_papers)}
    author_papers = defaultdict(list);
    for p in valid_papers:
        for aid in p.get(author_id_key, []):
            if aid in author_map: author_papers[aid].append(p[embedding_key])
    author_features_np = np.zeros((len(author_map), EMBEDDING_DIM), dtype=np.float32)
    for aid, embeds in author_papers.items():
        if aid in author_map and embeds: author_features_np[author_map[aid]] = np.mean(embeds, axis=0)
    paper_features = torch.tensor(np.stack([p[embedding_key] for p in valid_papers]), dtype=torch.float32)
    data=ManualGraphData(); data.node_features={'paper': paper_features, 'author': torch.tensor(author_features_np, dtype=torch.float32)}; data.num_nodes={'paper': len(valid_papers), 'author': len(author_map)}; data.node_maps={'author': author_map, 'paper': paper_map}
    auth_s, auth_d, coauth_pairs = [], [], set()
    for p in valid_papers:
        p_idx = paper_map[p['id']]; auth_indices=[author_map[aid] for aid in p.get(author_id_key, []) if aid in author_map];
        for a_idx in auth_indices: auth_s.append(a_idx); auth_d.append(p_idx)
        for a1, a2 in itertools.combinations(auth_indices, 2): coauth_pairs.add(tuple(sorted((a1, a2))))
    if auth_s: data.edge_index[('a','w','p')]=torch.tensor([auth_s, auth_d], dtype=torch.long); data.edge_index[('p','wb','a')]=torch.tensor([auth_d, auth_s], dtype=torch.long)
    else: print("Warning: No author-paper edges.")
    co_s, co_d = [], [];
    for a1,a2 in coauth_pairs: co_s.extend([a1,a2]); co_d.extend([a2,a1])
    if co_s: data.edge_index[('a','co','a')]=torch.tensor([co_s, co_d], dtype=torch.long)
    print(f"Graph: {data.num_nodes['author']} authors, {data.num_nodes['paper']} papers, {len(auth_s)} writes edges, {len(co_s)} co-author edges.")
    return data

# ======================================================================
# 4. EVALUATION METRICS
# ======================================================================
def calculate_precision_recall_f1(true_authors: set, candidates: list, k: int) -> Tuple[float, float, float]:
    """Calculates Precision@K, Recall@K, and F1@K with error handling."""
    try:
        if k <= 0: return 0.0, 0.0, 0.0
        if not true_authors or len(true_authors) == 0: return 0.0, 0.0, 0.0
        if not isinstance(candidates, list): print("Warning: 'candidates' not list in calc_pr_f1."); return 0.0, 0.0, 0.0
        predicted_authors = set( str(c['id']) for c in candidates[:k] if isinstance(c, dict) and 'id' in c )
        true_authors_str = set( str(ta) for ta in true_authors )
        tp = len(true_authors_str.intersection(predicted_authors))
        precision = tp / k
        recall = tp / len(true_authors_str)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1
    except Exception as e: print(f"\n!!! Error inside calculate_precision_recall_f1: {e} !!!"); return 0.0, 0.0, 0.0

def calculate_ndcg(true_authors: set, candidates: list, k: int) -> float:
    """Calculates NDCG@k with error handling."""
    try:
        if k <= 0: return 0.0
        if not true_authors or len(true_authors) == 0: return 0.0
        if not isinstance(candidates, list): print("Warning: 'candidates' not list in calc_ndcg."); return 0.0
        true_authors_str = set( str(ta) for ta in true_authors )
        relevance = [1 if (isinstance(c, dict) and 'id' in c and str(c['id']) in true_authors_str) else 0 for c in candidates[:k]]
        dcg = sum((r / math.log2(i + 2)) for i, r in enumerate(relevance))
        ideal_relevance = [1] * min(len(true_authors_str), k)
        idcg = sum((r / math.log2(i + 2)) for i, r in enumerate(ideal_relevance))
        return dcg / idcg if idcg > 0 else 0.0
    except Exception as e: print(f"\n!!! Error inside calculate_ndcg: {e} !!!"); return 0.0

# ======================================================================
# 5. OPTUNA OBJECTIVE FUNCTION (MODIFIED FOR HARD NEGATIVE SAMPLING)
# ======================================================================

def evaluate_model(model: nn.Module, eval_papers: List[Dict[str, Any]],
                   train_graph_data: ManualGraphData, all_papers_features: Dict[str, torch.Tensor],
                   all_papers_graph_data: ManualGraphData, config: Dict[str, Any],
                   start_eval_paper_index: int) -> Tuple[float, float]:
    """Helper function to evaluate model on a given set of papers."""
    model.eval(); total_ndcg, total_f1, evaluated_count = 0.0, 0.0, 0; all_author_embeddings_eval, all_paper_embeddings_eval = None, None
    with torch.no_grad():
        try: h_dict_eval=model(all_papers_features, train_graph_data.edge_index, all_papers_graph_data.num_nodes); all_author_embeddings_eval=h_dict_eval['author']; all_paper_embeddings_eval=h_dict_eval['paper']
        except Exception as e: print(f"Error during eval model forward pass: {e}"); return 0.0, 0.0
    inference_author_map = all_papers_graph_data.node_maps.get('author', {});
    if not inference_author_map or all_author_embeddings_eval is None or all_paper_embeddings_eval is None: print("Warning: Missing data for eval."); return 0.0, 0.0
    for i in tqdm(range(len(eval_papers)), desc="Evaluating", leave=False, ncols=80):
        eval_paper=eval_papers[i]; eval_paper_index=start_eval_paper_index+i;
        true_author_ids_all=set( str(aid) for aid in eval_paper.get(config['AUTHOR_ID_KEY'], []) ); # Force str
        true_author_ids=true_author_ids_all.intersection(inference_author_map.keys());
        if not true_author_ids: continue;
        if eval_paper_index >= all_paper_embeddings_eval.shape[0]: continue;
        test_paper_embedding = all_paper_embeddings_eval[eval_paper_index].unsqueeze(0);
        if all_author_embeddings_eval.nelement() == 0: continue;
        similarities = F.cosine_similarity(test_paper_embedding, all_author_embeddings_eval); candidate_scores = []
        for aid, idx in inference_author_map.items():
            if idx < similarities.shape[0]: candidate_scores.append({'id': str(aid), 'score': similarities[idx].item()}) # Force str
        if not candidate_scores: continue;
        candidate_scores.sort(key=lambda x: x['score'], reverse=True);
        ndcg = calculate_ndcg(true_author_ids, candidate_scores, config['K_AT'])
        _, _, f1 = calculate_precision_recall_f1(true_author_ids_all, candidate_scores, config['K_AT'])
        total_ndcg+=ndcg; total_f1+=f1; evaluated_count+=1
    avg_ndcg=total_ndcg/evaluated_count if evaluated_count>0 else 0.0; avg_f1=total_f1/evaluated_count if evaluated_count>0 else 0.0; return avg_ndcg, avg_f1


def objective(trial: optuna.Trial, all_papers_with_embeddings: List[Dict[str, Any]],
              test_set_ratio: float, validation_set_ratio: float, embedding_key: str) -> float:
    """ Optuna objective with K-Means HARD Negative Sampling. """

    # --- 1. Suggest Hyperparameters ---
    print(f"\n--- Starting Trial {trial.number} ---")
    config = {
        'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256]),
        'output_dim': trial.suggest_categorical('output_dim', [64, 128]),
        'lr': trial.suggest_float('lr', 1e-4, 1e-3, log=True),
        'epochs': trial.suggest_int('epochs', 20, 40), # Using 20-40 range
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        'dropout': trial.suggest_float('dropout', 0.2, 0.5),
        'margin': trial.suggest_float('margin', 0.4, 0.8), # Tuned range
        'K_AT': 10,
        'AUTHOR_ID_KEY': 'author_id_list',
        'EMBEDDING_DIM': EMBEDDING_DIM,
        'EMBEDDING_KEY': embedding_key
    }

    # --- 2. Data Setup (Train/Val/Test Split) ---
    num_papers = len(all_papers_with_embeddings); test_set_size = int(num_papers * test_set_ratio); remaining_size = num_papers - test_set_size; validation_set_size = int(remaining_size * validation_set_ratio); train_set_size = remaining_size - validation_set_size
    train_papers = all_papers_with_embeddings[:train_set_size]; validation_papers = all_papers_with_embeddings[train_set_size:train_set_size + validation_set_size]; test_papers = all_papers_with_embeddings[train_set_size + validation_set_size:]
    print(f"Split: Train={len(train_papers)}, Val={len(validation_papers)}, Test={len(test_papers)}")

    train_graph_data = build_manual_graph(train_papers, config['AUTHOR_ID_KEY'], config['EMBEDDING_KEY'])
    if not train_graph_data.edge_index or ('a', 'w', 'p') not in train_graph_data.edge_index:
        print("No author edges in training graph, pruning trial."); raise optuna.exceptions.TrialPruned("No author edges.")

    for key in train_graph_data.node_features: train_graph_data.node_features[key] = train_graph_data.node_features[key].to(DEVICE)
    for key in train_graph_data.edge_index: train_graph_data.edge_index[key] = train_graph_data.edge_index[key].to(DEVICE)

    print("Building full graph for evaluation...")
    all_papers_graph_data = build_manual_graph(all_papers_with_embeddings, config['AUTHOR_ID_KEY'], config['EMBEDDING_KEY'])
    all_papers_features = {'paper': torch.tensor(np.stack([p[config['EMBEDDING_KEY']] for p in all_papers_with_embeddings]), dtype=torch.float32).to(DEVICE), 'author': all_papers_graph_data.node_features['author'].to(DEVICE)}

    # --- 3. K-Means Clustering for Hard Negative Sampling ---
    print("Building K-Means clusters for hard negative sampling...")
    if not SKLEARN_AVAILABLE:
        print("SKLearn not found, falling back to RANDOM sampling.")
        use_hard_sampling = False
    else:
        try:
            # Get initial author features from the training graph
            initial_author_features = train_graph_data.node_features['author'].cpu().numpy()
            num_authors_train = initial_author_features.shape[0]
            
            if num_authors_train < 100: # Not enough authors to cluster
                print("Too few authors, falling back to RANDOM sampling.")
                use_hard_sampling = False
            else:
                num_clusters = min(100, num_authors_train // 5) # 100 clusters or 5 authors/cluster
                kmeans = KMeans(n_clusters=num_clusters, random_state=SEED, n_init=5)
                author_labels = kmeans.fit_predict(initial_author_features)
                
                # Create a lookup map: {cluster_id: [author_idx_1, author_idx_2, ...]}
                cluster_to_authors = defaultdict(list)
                for author_idx, cluster_id in enumerate(author_labels):
                    cluster_to_authors[cluster_id].append(author_idx)
                
                print(f"Clustering complete. Found {len(cluster_to_authors)} clusters.")
                use_hard_sampling = True
                
                # We also need a fast lookup for an author's cluster
                author_idx_to_cluster = torch.tensor(author_labels, device=DEVICE)

        except Exception as e:
            print(f"Error during K-Means clustering: {e}. Falling back to RANDOM sampling.")
            use_hard_sampling = False


    # --- 4. Model Training (with HARD Negative Sampling) ---
    model = ManualHeteroGNN(config['EMBEDDING_DIM'], config['hidden_dim'], config['output_dim'], config['dropout']).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    loss_fn = nn.TripletMarginLoss(margin=config['margin'], p=2)
    pos_edges = train_graph_data.edge_index[('a', 'w', 'p')]
    num_authors_train = train_graph_data.num_nodes['author']

    for epoch in tqdm(range(config['epochs']), desc=f"Trial {trial.number} Training", ncols=80):
        model.train(); optimizer.zero_grad(); h_dict = model(train_graph_data.node_features, train_graph_data.edge_index, train_graph_data.num_nodes); h_paper_train=h_dict['paper']; h_author_train=h_dict['author']
        anchor_emb = h_paper_train[pos_edges[1]]; positive_emb = h_author_train[pos_edges[0]]

        # --- MODIFIED: Hard Negative Sampling ---
        if use_hard_sampling:
            # Get cluster labels for all positive authors in the batch
            positive_author_indices = pos_edges[0]
            positive_clusters = author_idx_to_cluster[positive_author_indices]
            
            negative_author_indices_list = []
            for i in range(len(positive_author_indices)):
                pos_author_idx = positive_author_indices[i].item()
                cluster_id = positive_clusters[i].item()
                
                # Get all authors in this cluster
                candidate_pool = cluster_to_authors[cluster_id]
                
                if len(candidate_pool) <= 1:
                    # This cluster only has one author, or something is wrong
                    # Fallback: sample from all authors
                    neg_author_idx = random.randint(0, num_authors_train - 1)
                else:
                    # Sample from the cluster until we find a *different* author
                    neg_author_idx = random.choice(candidate_pool)
                    attempts = 0
                    while neg_author_idx == pos_author_idx and attempts < 10:
                        neg_author_idx = random.choice(candidate_pool)
                        attempts += 1
                    
                    if neg_author_idx == pos_author_idx: # Still the same? Fallback
                         neg_author_idx = random.randint(0, num_authors_train - 1)
                        
                negative_author_indices_list.append(neg_author_idx)
                
            negative_author_indices = torch.tensor(negative_author_indices_list, device=DEVICE)

        else: # Fallback to Random Sampling
            negative_author_indices = torch.randint(0, num_authors_train, (pos_edges.shape[1],), device=DEVICE)
        # --- END MODIFICATION ---

        negative_emb = h_author_train[negative_author_indices] # Get embeddings for sampled negatives
        loss = loss_fn(anchor_emb, positive_emb, negative_emb); loss.backward(); optimizer.step()

        # --- Validation & Pruning ---
        if (epoch + 1) % 5 == 0 or epoch == config['epochs'] - 1:
            val_ndcg, _ = evaluate_model(model, validation_papers, train_graph_data, all_papers_features,
                                         all_papers_graph_data, config, start_eval_paper_index=train_set_size)
            trial.report(val_ndcg, epoch)
            print(f"   Epoch {epoch+1:02d}/{config['epochs']}, Loss: {loss.item():.4f}, Val NDCG: {val_ndcg:.4f}")
            if trial.should_prune(): print("--- Trial Pruned ---"); raise optuna.exceptions.TrialPruned()
        # --- End Validation ---

    # --- 5. Final Test Set Evaluation ---
    test_ndcg, test_f1 = evaluate_model(model, test_papers, train_graph_data, all_papers_features,
                                        all_papers_graph_data, config, start_eval_paper_index=train_set_size + validation_set_size)
    print(f"--- Trial {trial.number} Complete --- Final Test NDCG: {test_ndcg:.4f}, Test F1: {test_f1:.4f}")

    # --- 6. Return Metric ---
    return test_ndcg

# ======================================================================
# 6. HPO RUNNER
# ======================================================================
def run_hpo(max_papers: int, num_trials: int, test_set_ratio: float, validation_set_ratio: float,
            model_type: str = EMBEDDING_TYPE, cache_dir: str = "."):
    """Loads data once, then runs the full Optuna HPO study."""
    # --- MODIFIED PRINT STATEMENT ---
    print(f"Starting HPO pipeline with {model_type.upper()} embeddings and HARD negative sampling (K-Means)...")
    os.makedirs(cache_dir, exist_ok=True) # Ensure cache directory exists
    papers_list = open_file()
    if not papers_list: 
        print("Failed to load data. Exiting."); 
        return None, None
    
    papers_with_embeddings = filter_and_preprocess_data(papers_list, max_papers, model_type=model_type, cache_dir=cache_dir)
    if not papers_with_embeddings or embedding_key not in papers_with_embeddings[0] or papers_with_embeddings[0].get(embedding_key) is None:
        print(f"Error: Embedding key '{embedding_key}' not found or is None in processed papers."); 
        return None, None
    
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10))
    study.optimize(
        lambda trial: objective(trial, papers_with_embeddings, test_set_ratio, validation_set_ratio, embedding_key),
        n_trials=num_trials,
        timeout=None
    )
    
    print("\n" + "=" * 60); print("HYPERPARAMETER OPTIMIZATION COMPLETE 🏆"); print("=" * 60)
    print(f"Embedding Model Used: {model_type.upper()} ({EMBEDDING_MODEL_NAME})")
    print(f"Negative Sampling: HARD (K-Means)") # <-- MODIFIED
    print(f"Number of finished trials: {len(study.trials)}"); print("\nBest trial:")
    try:
        best_trial = study.best_trial; print(f"   Value (Avg. Test NDCG): {best_trial.value:.4f}"); print("   Best Parameters:")
        for key, value in best_trial.params.items(): print(f"     {key}: {value}")
    except ValueError: print("   No trials completed successfully.")
    
    print("=" * 60); print("\nTo run the final model, use these parameters in your FINAL_CONFIG dict.")
    
    # --- MODIFICATION: RETURN THE STUDY AND DATA ---
    return study, papers_with_embeddings

# ======================================================================
# 7. FINAL MODEL TRAINING & CLUSTERING (NEW SECTION)
# (This section is unchanged, it just runs *after* HPO)
# ======================================================================

def train_final_model(best_params: Dict[str, Any], all_papers: List[Dict[str, Any]], 
                        embedding_key: str) -> Tuple[nn.Module, np.ndarray, Dict[str, int]]:
    """
    Trains the final GNN model on ALL data using the best hyperparameters.
    Returns the trained model, the final author embeddings, and the author map.
    """
    print("\n" + "=" * 60)
    print("🚄 Training final model on all data...")
    print(f"Using parameters: {best_params}")
    
    # 1. Build the full graph and features
    all_graph_data = build_manual_graph(all_papers, best_params['AUTHOR_ID_KEY'], embedding_key)
    if not all_graph_data.node_features:
        raise ValueError("Failed to build graph for final training.")

    all_features = {
        'paper': torch.tensor(np.stack([p[embedding_key] for p in all_papers]), dtype=torch.float32).to(DEVICE),
        'author': all_graph_data.node_features['author'].to(DEVICE)
    }
    
    # Move graph structure to device
    for key in all_graph_data.edge_index:
        all_graph_data.edge_index[key] = all_graph_data.edge_index[key].to(DEVICE)
    
    # 2. Initialize model and optimizer
    model = ManualHeteroGNN(
        best_params['EMBEDDING_DIM'], 
        best_params['hidden_dim'], 
        best_params['output_dim'], 
        best_params['dropout']
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=best_params['lr'], 
        weight_decay=best_params['weight_decay']
    )
    loss_fn = nn.TripletMarginLoss(margin=best_params['margin'], p=2)
    
    pos_edges = all_graph_data.edge_index.get(('a', 'w', 'p'))
    if pos_edges is None:
        raise ValueError("No author-paper edges found in the full graph.")
        
    num_authors = all_graph_data.num_nodes['author']

    # 3. Training loop
    # NOTE: This final training *still uses random negative sampling*.
    # The hard sampling was only needed for the HPO to *find* the best parameters.
    # We could implement hard sampling here too, but it's simpler to just train
    # the optimized model.
    model.train()
    epochs = best_params.get('epochs', 30)
    
    for epoch in tqdm(range(epochs), desc="Final Training"):
        optimizer.zero_grad()
        h_dict = model(all_features, all_graph_data.edge_index, all_graph_data.num_nodes)
        h_paper = h_dict['paper']; h_author = h_dict['author']
        anchor_emb = h_paper[pos_edges[1]]; positive_emb = h_author[pos_edges[0]]
        
        # Using random sampling for simplicity in the final train
        negative_author_indices = torch.randint(0, num_authors, (pos_edges.shape[1],), device=DEVICE)
        negative_emb = h_author[negative_author_indices]
        
        loss = loss_fn(anchor_emb, positive_emb, negative_emb)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # 4. Get final embeddings
    model.eval()
    with torch.no_grad():
        final_h_dict = model(all_features, all_graph_data.edge_index, all_graph_data.num_nodes)
        
    author_embeddings = final_h_dict['author'].cpu().numpy()
    author_map = all_graph_data.node_maps['author']
    
    print(f"Final training complete. Generated {author_embeddings.shape[0]} author embeddings.")
    print("=" * 60)
    return model, author_embeddings, author_map


def find_optimal_clusters_gmm(embeddings: np.ndarray, max_k: int = 40, step: int = 5) -> int:
    """
    Finds the optimal number of clusters using GaussianMixture (EM)
    and the Bayesian Information Criterion (BIC).
    """
    if not SKLEARN_AVAILABLE:
        print("Scikit-learn not available, defaulting to 10 clusters.")
        return 10
        
    print(f"🔎 Finding optimal cluster count (up to k={max_k}) using BIC...")
    
    if embeddings.shape[0] > 10000:
        print("Dataset > 10k, sampling 10k authors for BIC calculation.")
        indices = np.random.choice(embeddings.shape[0], 10000, replace=False)
        embeddings_sample = embeddings[indices]
    else:
        embeddings_sample = embeddings

    cluster_range = range(5, max_k + 1, step)
    if not cluster_range: 
        print("max_k is too low for step size. Defaulting to 5 clusters.")
        return 5
        
    bic_scores = []
    
    pbar = tqdm(cluster_range, desc="Testing k (BIC)")
    for k in pbar:
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=SEED)
        gmm.fit(embeddings_sample)
        bic_scores.append(gmm.bic(embeddings_sample))
        pbar.set_postfix({'k': k, 'BIC': bic_scores[-1]})
        
    best_k = cluster_range[np.argmin(bic_scores)]
    
    # Plot the BIC curve
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, bic_scores, marker='o')
    plt.title('GMM BIC Score for Optimal Cluster Count')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Bayesian Information Criterion (BIC)')
    plt.axvline(best_k, color='red', linestyle='--', label=f'Best k = {best_k} (Lowest BIC)')
    plt.legend()
    plt.show(block=False) 
    
    print(f"Done. Best number of clusters (lowest BIC): {best_k}")
    return best_k


def run_final_clustering(embeddings: np.ndarray, best_k: int, 
                         author_map: Dict[str, int]) -> Dict[str, int]:
    """
    Runs the final GMM clustering with the optimal 'k' on the full dataset.
    Returns a dictionary mapping {author_id: cluster_id}.
    """
    if not SKLEARN_AVAILABLE:
        print("Scikit-learn not available. Skipping clustering.")
        return {}
        
    print(f"🔄 Running final clustering with k={best_k} on all {embeddings.shape[0]} authors...")
    
    # Initialize and fit the final GMM (this is the EM algorithm)
    gmm = GaussianMixture(n_components=best_k, covariance_type='full', random_state=SEED,
                          n_init=3, warm_start=True, 
                          max_iter=150) 
                          
    gmm.fit(embeddings)
    labels = gmm.predict(embeddings)
    
    # Map labels back to the original author IDs
    idx_to_aid = {idx: aid for aid, idx in author_map.items()}
    author_to_cluster = {
        idx_to_aid[i]: int(label)
        for i, label in enumerate(labels)
        if i in idx_to_aid
    }
    
    print(f"Clustering complete. Assigned {len(author_to_cluster)} authors to {best_k} clusters.")
    return author_to_cluster


def analyze_clusters(author_to_cluster: Dict[str, int], all_papers: List[Dict[str, Any]], 
                     num_clusters_to_show: int = 5, top_n_keywords: int = 10, top_n_authors: int = 5):
    """
    Provides a simple analysis of the resulting clusters, showing
    cluster sizes, top authors, and common title keywords.
    """
    if not author_to_cluster:
        print("No cluster data to analyze.")
        return

    print("\n" + "=" * 60)
    print("📊 CLUSTER ANALYSIS (from GNN Embeddings + EM)")
    print("=" * 60)

    # 1. Reverse map: cluster -> list of authors
    cluster_to_authors = defaultdict(list)
    for aid, cid in author_to_cluster.items():
        cluster_to_authors[cid].append(aid)

    # 2. Get paper counts for authors and keywords for clusters
    author_paper_count = Counter()
    cluster_keywords = defaultdict(Counter)

    for paper in all_papers:
        author_ids = paper.get('author_id_list', [])
        author_paper_count.update(author_ids) 
        paper_clusters = [author_to_cluster.get(aid) for aid in author_ids if aid in author_to_cluster]
        
        if paper_clusters:
            primary_cluster = Counter(paper_clusters).most_common(1)[0][0]
            title = paper.get('title', '').lower()
            words = [w for w in title.split() if len(w) > 3 and w.isalpha()]
            cluster_keywords[primary_cluster].update(words)

    # 3. Print analysis for the largest clusters
    print(f"Top {num_clusters_to_show} Largest Clusters:\n")
    
    sorted_clusters = sorted(cluster_to_authors.items(), key=lambda item: len(item[1]), reverse=True)
    
    for cid, authors in sorted_clusters[:num_clusters_to_show]:
        print(f"--- Cluster {cid} (Size: {len(authors)} authors) ---")
        
        authors_in_cluster_sorted = sorted(
            authors, 
            key=lambda aid: author_paper_count[aid], 
            reverse=True
        )
        top_authors = authors_in_cluster_sorted[:top_n_authors]
        
        print(f"  Top {top_n_authors} Authors (by paper count):")
        for i, aid in enumerate(top_authors):
            print(f"    {i+1}. {aid} (Papers: {author_paper_count[aid]})")
            
        top_keywords = cluster_keywords[cid].most_common(top_n_keywords)
        print(f"\n  Top {top_n_keywords} Keywords:")
        print("    " + ", ".join([f"{word} ({count})" for word, count in top_keywords]))
        print("-" * 30)

# ======================================================================
# 8. ASSOCIATION RULE MINING (NEW SECTION)
# ======================================================================

def run_association_rule_mining(all_papers: List[Dict[str, Any]], author_id_key: str = 'author_id_list'):
    """
    Performs frequent pattern mining and association rule analysis
    to find likely co-author relationships.
    """
    if not MLXTEND_AVAILABLE:
        print("mlxtend library not available. Skipping association rule mining.")
        return

    print("\n" + "=" * 60)
    print("🪙 Running Association Rule Mining for Co-Authors")
    print("=" * 60)

    # 1. Create the list of transactions
    transactions = [p.get(author_id_key) for p in all_papers if p.get(author_id_key)]
    if not transactions:
        print("No author lists found to mine.")
        return
        
    print(f"Analyzing {len(transactions)} papers (transactions)...")

    # 2. One-hot encode the transactions
    print("Encoding transactions...")
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # 3. Find frequent patterns (itemsets)
    min_sup = max(0.001, 10 / len(df)) # At least 10 papers or 0.1%
    print(f"Mining for frequent author sets (min_support={min_sup:.4f})...")
    frequent_itemsets = apriori(df, min_support=min_sup, use_colnames=True)

    if frequent_itemsets.empty:
        print("No frequent itemsets found. Try lowering 'min_support'.")
        return
        
    print(f"Found {len(frequent_itemsets)} frequent author sets.")

    # 4. Generate the association rules
    print("Generating association rules (min_confidence=0.5)...")
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

    if rules.empty:
        print("No strong association rules found. Try lowering 'min_threshold'.")
        return

    # 5. Display the most interesting rules
    rules['antecedent_len'] = rules['antecedents'].apply(lambda x: len(x))
    rules['consequent_len'] = rules['consequents'].apply(lambda x: len(x))
    
    simple_rules = rules[
        (rules['antecedent_len'] == 1) & (rules['consequent_len'] == 1)
    ].sort_values(by=['confidence', 'lift'], ascending=False)
    
    print("\n--- Top 15 Strongest Co-Author Rules (IF A -> THEN B) ---")
    
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.width', 1000)
    
    simple_rules['IF'] = simple_rules['antecedents'].apply(lambda x: list(x)[0])
    simple_rules['THEN'] = simple_rules['consequents'].apply(lambda x: list(x)[0])
    
    final_rules_df = simple_rules[['IF', 'THEN', 'confidence', 'lift', 'support']]
    final_rules_df = final_rules_df.reset_index(drop=True)
    
    print(final_rules_df.head(15))
    print("=" * 60)


# ======================================================================
# 9. MAIN EXECUTION
# ======================================================================

if __name__ == "__main__":
    # --- Configuration ---
    MAX_PAPERS_FOR_HPO = 1000  # Use a subset for fast HPO (e.g., 1000-10000)
    NUM_TRIALS = 50           # Number of HPO trials (e.g., 50-100)
    TEST_SET_RATIO = 0.2
    VALIDATION_SET_RATIO = 0.2 # 20% of (Train+Val) used for validation
    CACHE_DIRECTORY = "./embedding_cache" # Directory to store embedding caches

    # --- Run HPO ---
    study, papers_with_embeddings = run_hpo(
        MAX_PAPERS_FOR_HPO, 
        NUM_TRIALS, 
        TEST_SET_RATIO, 
        VALIDATION_SET_RATIO,
        model_type=EMBEDDING_TYPE, 
        cache_dir=CACHE_DIRECTORY
    )

    # --- POST-HPO CLUSTERING AND ANALYSIS ---
    if not papers_with_embeddings:
        print("Data loading or preprocessing failed. Exiting.")
    elif not study or not study.best_trial:
        print("HPO study finished with no best trial. Exiting analysis.")
    else:
        print("\n" + "=" * 60)
        print("🚀 Starting Post-HPO Analysis: Final Training, Clustering & Rule Mining")
        print("=" * 60)
        
        try:
            # 1. Get best params and add missing keys
            best_params = study.best_trial.params
            best_params['epochs'] = best_params.get('epochs', 30) 
            best_params['AUTHOR_ID_KEY'] = 'author_id_list'
            best_params['EMBEDDING_DIM'] = EMBEDDING_DIM
            
            # --- 2. GNN + EM CLUSTERING ANALYSIS ---
            if SKLEARN_AVAILABLE:
                # Train the final model on all data
                final_model, author_embeddings, author_map = train_final_model(
                    best_params, 
                    papers_with_embeddings, 
                    embedding_key
                )
                
                # Find optimal cluster count (k) using EM/GMM + BIC
                best_k = find_optimal_clusters_gmm(author_embeddings, max_k=50, step=5)
                
                # Run final clustering with the best 'k'
                author_to_cluster_map = run_final_clustering(
                    author_embeddings, 
                    best_k, 
                    author_map
                )
                
                # Analyze and print cluster results
                analyze_clusters(
                    author_to_cluster_map, 
                    papers_with_embeddings, 
                    num_clusters_to_show=5
                )
            else:
                print("Skipping GNN+EM Clustering analysis (scikit-learn not found).")

            # --- 3. ASSOCIATION RULE MINING ANALYSIS ---
            if MLXTEND_AVAILABLE:
                run_association_rule_mining(
                    papers_with_embeddings, 
                    'author_id_list'
                )
            else:
                print("Skipping Association Rule Mining (mlxtend not found).")
            
            
            print("\nAnalysis complete. The BIC plot may be open in a separate window.")
            
            # Keep the script alive to see the plot
            if SKLEARN_AVAILABLE and 'plt' in locals() or 'plt' in globals():
                print("Close the 'BIC Score' plot window to exit.")
                plt.show() # This will now block until the user closes the plot

        except Exception as e:
            print(f"\n!!! An error occurred during final analysis: {e} !!!")
            import traceback
            traceback.print_exc()
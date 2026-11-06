import numpy as np
import time
import json
import math
import warnings
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import itertools
import os
import pickle
import random
from tqdm import tqdm # <-- For progress bars

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Transformers imports
# Optional if using sentence-transformers for SPECTER
# from transformers import AutoTokenizer, AutoModel

# *** Import SentenceTransformer if using SPECTER ***
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    print("Warning: sentence-transformers library not found. SPECTER embedding generation will not work.")
    print("Install it via: pip install sentence-transformers")

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
# 1. EMBEDDING MODEL INITIALIZATION (SciBERT or SPECTER)
# ======================================================================
# --- CHOOSE YOUR EMBEDDING MODEL ---
EMBEDDING_MODEL_NAME = 'allenai/specter' # Or 'allenai/scibert_scivocab_uncased'
EMBEDDING_TYPE = 'specter'           # Or 'scibert'
# ---

EMBEDDING_DIM = 768
embedding_key = f'{EMBEDDING_TYPE}_embedding' # Global embedding key based on type

def initialize_embedding_model(model_name: str, model_type: str) -> Tuple[Optional[Any], Optional[Any], torch.device]:
    """Initializes either SciBERT or a SentenceTransformer model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = None
    model = None
    if model_type == 'scibert':
        try:
            from transformers import AutoTokenizer, AutoModel # Local import
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.to(device)
            print(f"SciBERT Model loaded on: {device} (Model: {model_name})")
        except Exception as e: print(f"Error loading SciBERT model: {e}")
    elif model_type == 'specter':
        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                model = SentenceTransformer(model_name, device=str(device))
                print(f"SentenceTransformer Model loaded on: {device} (Model: {model_name})")
            except Exception as e: print(f"Error loading SentenceTransformer model: {e}")
        else: print("Cannot load SPECTER model, sentence-transformers not installed.")
    else: print(f"Unknown model_type: {model_type}.")
    return tokenizer, model, device

embed_tokenizer, embed_model, DEVICE = initialize_embedding_model(EMBEDDING_MODEL_NAME, EMBEDDING_TYPE)

def get_embedding(text_list: List[str], model_type: str) -> np.ndarray:
    """Generates embeddings using the chosen model."""
    if not text_list or embed_model is None: return np.random.rand(len(text_list), EMBEDDING_DIM).astype(np.float32)
    try:
        if model_type == 'scibert':
            inputs = embed_tokenizer( text_list, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad(): outputs = embed_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        elif model_type == 'specter' and SENTENCE_TRANSFORMER_AVAILABLE:
             embeddings = embed_model.encode(text_list, convert_to_numpy=True, show_progress_bar=False)
        else: raise ValueError("Model not loaded or unknown type")
        return embeddings.astype(np.float32)
    except Exception as e:
        print(f"Error generating embeddings with {model_type}: {e}")
        return np.random.rand(len(text_list), EMBEDDING_DIM).astype(np.float32)

# ======================================================================
# 2. DATA PREPROCESSING AND LOADING (with Caching & Tracing)
# ======================================================================
def preprocess_papers_with_embeddings(papers: List[Dict[str, Any]], model_type: str, cache_path: str) -> List[Dict[str, Any]]:
    """Generates embeddings (SciBERT or SPECTER) for all papers, using caching."""
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
        # Ensure correct key exists, even if loaded from old cache/data
        for paper in papers:
            if current_embedding_key not in paper:
                found_key=None;
                for key in paper:
                    if 'embedding' in key: paper[current_embedding_key]=paper.pop(key); found_key=key; break
                if not found_key: paper[current_embedding_key]=None
        papers = [p for p in papers if p.get(current_embedding_key) is not None]
        return papers

    print(f"Generating {model_type} embeddings for {len(papers_requiring_embedding)} papers...")
    texts = []
    # --- Print Text for First and Last Paper to be Embedded ---
    trace_indices = []
    if papers_requiring_embedding:
        trace_indices.append(0) # First paper index in this batch
        if len(papers_requiring_embedding) > 1:
            trace_indices.append(len(papers_requiring_embedding) - 1) # Last paper index

    for i, p in enumerate(papers_requiring_embedding):
        title = p.get('title', '')
        if model_type == 'specter':
             # Format for SPECTER
             combined_text = f"{title}{embed_model.tokenizer.sep_token}{p.get('abstract', '')}"
        else: # SciBERT format
            abstract = p.get('abstract', '')
            venue = p.get('venue', '')
            year = str(p.get('year', ''))
            combined_text = f"{title} [SEP] {venue} [SEP] {year} [SEP] {abstract}"
        texts.append(combined_text)
        # Print if it's one of the trace indices
        if i in trace_indices:
             print(f"\n--- Tracing Paper ID: {p.get('id')} (Index {i} in embedding batch) ---")
             print(f"Combined Text for Embedding:\n>>>\n{combined_text}\n<<<\n")
    # --- End Trace ---

    all_gen_embeddings = []
    batch_size=32
    for i in tqdm(range(0, len(texts), batch_size), desc=f"{model_type.capitalize()} Embedding Batches"):
        batch_texts = texts[i:i+batch_size]
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
    # (Same as before)
    try:
        with open(filename, 'r', encoding='utf-8') as f: raw_data = json.load(f)
        papers_list = list(raw_data.values())
        print(f"Successfully loaded {len(papers_list)} papers from {filename}")
        for i, p in enumerate(papers_list):
            if 'id' not in p or p['id'] is None: p['id'] = f"paper_{i}_{time.time()}"
        return papers_list
    except FileNotFoundError:
        print(f"File {filename} not found. Generating dummy data...")
        return generate_dummy_data(1000) # Default size if file missing
    except Exception as e: print(f"Error loading file: {e}"); return None

def generate_dummy_data(num_papers: int) -> List[Dict[str, Any]]:
    # (Same as before)
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
    # (Same as before)
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
    # The actual embedding + text printing happens inside here now
    return preprocess_papers_with_embeddings(valid_papers, model_type=model_type, cache_path=cache_path)

# ======================================================================
# 3. GRAPH AND GNN IMPLEMENTATION (GCN + Co-Authors)
# ======================================================================
# --- ManualGraphData, ManualGCNLayer, ManualHeteroGNN classes ---
class ManualGraphData:
    def __init__(self): self.node_features={}; self.edge_index={}; self.node_maps={}; self.num_nodes={}
class ManualGCNLayer(nn.Module):
    def __init__(self, i, o, d): super().__init__(); self.lin=nn.Linear(i,o); self.drop=nn.Dropout(d)
    def forward(self, x, ei, n): s,d=ei; a=torch.zeros(n, x.size(1), dtype=x.dtype, device=x.device); a.index_add_(0,d,x[s]); deg=torch.bincount(d, minlength=n).float().unsqueeze(1).clamp(min=1).to(x.device); return F.relu(self.lin(self.drop(a/deg)))
class ManualHeteroGNN(nn.Module):
    def __init__(self, pi, h, o, d): super().__init__(); self.lp=nn.Linear(pi,h); self.la=nn.Linear(pi,h); self.c_ap=ManualGCNLayer(h,h,d); self.c_pa=ManualGCNLayer(h,h,d); self.c_co=ManualGCNLayer(h,h,d); self.olp=nn.Linear(h,o); self.ola=nn.Linear(h,o)
    def forward(self, xd, eid, nnd):
        hp_i=F.relu(self.lp(xd['paper'])); ha_i=F.relu(self.la(xd['author'])); hp,ha=hp_i,ha_i; mp=torch.zeros_like(hp); ma=torch.zeros_like(ha)
        # Correct indentation
        if ('a','w','p') in eid: mp+=self.c_ap(ha, eid[('a','w','p')], nnd['paper'])
        if ('p','wb','a') in eid: ma+=self.c_pa(hp, eid[('p','wb','a')], nnd['author'])
        if ('a','co','a') in eid: ma+=self.c_co(ha, eid[('a','co','a')], nnd['author'])
        hp_f=hp_i+mp; ha_f=ha_i+ma; op=self.olp(hp_f); oa=self.ola(ha_f); return {'paper':op, 'author':oa}

def build_manual_graph(papers: List[Dict[str, Any]], author_id_key: str, embedding_key: str) -> ManualGraphData:
    """Builds the graph, using the specified embedding key."""
    all_author_ids = set(aid for p in papers for aid in p.get(author_id_key, []))
    if not all_author_ids or not papers: return ManualGraphData()
    author_map = {aid: idx for idx, aid in enumerate(all_author_ids)}
    valid_papers = [p for p in papers if p.get('id') is not None and embedding_key in p and p[embedding_key] is not None]
    if not valid_papers: print("Error: No valid papers for graph."); return ManualGraphData()
    paper_map = {p['id']: i for i, p in enumerate(valid_papers)}
    author_papers = defaultdict(list);
    for p in valid_papers:
        for aid in p.get(author_id_key, []):
            if aid in author_map: # Ensure author is in the current graph's map
                author_papers[aid].append(p[embedding_key])
    author_features_np = np.zeros((len(author_map), EMBEDDING_DIM), dtype=np.float32)
    for aid, embeds in author_papers.items():
        if aid in author_map and embeds: # Check again
            author_features_np[author_map[aid]] = np.mean(embeds, axis=0)
    paper_features = torch.tensor(np.stack([p[embedding_key] for p in valid_papers]), dtype=torch.float32)
    data=ManualGraphData(); data.node_features={'paper': paper_features, 'author': torch.tensor(author_features_np, dtype=torch.float32)}; data.num_nodes={'paper': len(valid_papers), 'author': len(author_map)}; data.node_maps={'author': author_map, 'paper': paper_map}
    auth_s, auth_d, coauth_pairs = [], [], set()
    for p in valid_papers:
        p_idx = paper_map[p['id']]
        auth_indices=[author_map[aid] for aid in p.get(author_id_key, []) if aid in author_map] # Ensure authors are in map
        # Correct indentation
        for a_idx in auth_indices:
            auth_s.append(a_idx)
            auth_d.append(p_idx)
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
# (Implementations with internal error handling)
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
    except Exception as e:
        print(f"\n!!! Error inside calculate_precision_recall_f1: {e} !!!"); return 0.0, 0.0, 0.0

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
    except Exception as e:
         print(f"\n!!! Error inside calculate_ndcg: {e} !!!"); return 0.0

# ======================================================================
# 5. FINAL MODEL TRAINING AND EVALUATION (with Hard Negatives & Tracing)
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
        # Use robust metric functions
        ndcg = calculate_ndcg(true_author_ids, candidate_scores, config['K_AT'])
        _, _, f1 = calculate_precision_recall_f1(true_author_ids_all, candidate_scores, config['K_AT'])
        total_ndcg+=ndcg; total_f1+=f1; evaluated_count+=1
    avg_ndcg=total_ndcg/evaluated_count if evaluated_count>0 else 0.0; avg_f1=total_f1/evaluated_count if evaluated_count>0 else 0.0; return avg_ndcg, avg_f1


def train_and_evaluate_final_model(all_papers_with_embeddings: List[Dict[str, Any]], config: Dict[str, Any], test_set_ratio: float, validation_set_ratio: float):
    """Trains and evaluates a single, final model using the best hyperparameters."""

    # --- 1. Data Setup ---
    num_papers = len(all_papers_with_embeddings); test_set_size = int(num_papers * test_set_ratio); remaining_size = num_papers - test_set_size; validation_set_size = int(remaining_size * validation_set_ratio); train_set_size = remaining_size - validation_set_size
    train_papers = all_papers_with_embeddings[:train_set_size]; validation_papers = all_papers_with_embeddings[train_set_size:train_set_size + validation_set_size]; test_papers = all_papers_with_embeddings[train_set_size + validation_set_size:]
    print(f"\nSplit: Train={len(train_papers)}, Val={len(validation_papers)}, Test={len(test_papers)}")

    embedding_key = config['EMBEDDING_KEY']
    train_graph_data = build_manual_graph(train_papers, config['AUTHOR_ID_KEY'], embedding_key)
    if not train_graph_data.edge_index or ('a', 'w', 'p') not in train_graph_data.edge_index: print("Error: No author edges in training graph."); return

    for key in train_graph_data.node_features: train_graph_data.node_features[key] = train_graph_data.node_features[key].to(DEVICE)
    for key in train_graph_data.edge_index: train_graph_data.edge_index[key] = train_graph_data.edge_index[key].to(DEVICE)

    print("Building full graph for evaluation...")
    all_papers_graph_data = build_manual_graph(all_papers_with_embeddings, config['AUTHOR_ID_KEY'], embedding_key)
    all_papers_features = {'paper': torch.tensor(np.stack([p[embedding_key] for p in all_papers_with_embeddings]), dtype=torch.float32).to(DEVICE), 'author': all_papers_graph_data.node_features['author'].to(DEVICE)}

    # --- Trace First and Last Paper Initial Embeddings ---
    if all_papers_with_embeddings: # Ensure list is not empty
        first_paper_trace = all_papers_with_embeddings[0]; last_paper_trace = all_papers_with_embeddings[-1]; first_paper_emb = first_paper_trace.get(embedding_key); last_paper_emb = last_paper_trace.get(embedding_key)
        print(f"\n--- Tracing First Paper ID: {first_paper_trace.get('id')} ---")
        if first_paper_emb is not None: print(f"Initial {EMBEDDING_TYPE.upper()} Embedding (first 10 dims):\n{first_paper_emb[:10]}...\nShape: {first_paper_emb.shape}\n")
        else: print("Initial embedding not found.")
        print(f"\n--- Tracing Last Paper ID: {last_paper_trace.get('id')} ---")
        if last_paper_emb is not None: print(f"Initial {EMBEDDING_TYPE.upper()} Embedding (first 10 dims):\n{last_paper_emb[:10]}...\nShape: {last_paper_emb.shape}\n")
        else: print("Initial embedding not found.")
    else: print("Warning: No papers available for tracing initial embeddings.")
    # --- End Trace ---

    # --- 2. Model Training ---
    print("\n--- Starting Final Model Training ---")
    model = ManualHeteroGNN(config['EMBEDDING_DIM'], config['hidden_dim'], config['output_dim'], config['dropout']).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    loss_fn = nn.TripletMarginLoss(margin=config['margin'], p=2)
    pos_edges = train_graph_data.edge_index[('a', 'w', 'p')]; num_authors_train = train_graph_data.num_nodes['author']; hard_negative_k = config['hard_negative_k']

    for epoch in tqdm(range(config['epochs']), desc="Final Training", ncols=80):
        model.train(); optimizer.zero_grad(); h_dict = model(train_graph_data.node_features, train_graph_data.edge_index, train_graph_data.num_nodes); h_paper_train=h_dict['paper']; h_author_train=h_dict['author']
        anchor_emb = h_paper_train[pos_edges[1]]; positive_emb = h_author_train[pos_edges[0]]
        # --- Top-K Hard Negative Sampling ---
        with torch.no_grad():
            sim_matrix=torch.matmul(anchor_emb, h_author_train.t()); mask=torch.zeros_like(sim_matrix, dtype=torch.bool); edge_indices=torch.arange(pos_edges.shape[1], device=DEVICE); true_author_indices=pos_edges[0]; mask[edge_indices, true_author_indices]=True; sim_matrix.masked_fill_(mask, -float('inf'))
            effective_k=min(hard_negative_k, sim_matrix.shape[1]-1) # Ensure k is valid
            if effective_k<=0 or num_authors_train <= 1: # Fallback if k=0 or only 1 author
                negative_author_indices=torch.randint(0, num_authors_train, (pos_edges.shape[1],), device=DEVICE)
            else:
                _,top_k_indices=torch.topk(sim_matrix, k=effective_k, dim=1); rand_choice=torch.randint(0, effective_k, (top_k_indices.shape[0],), device=DEVICE); negative_author_indices=top_k_indices[torch.arange(top_k_indices.shape[0]), rand_choice]
        negative_emb = h_author_train[negative_author_indices]
        # --- End Hard Negative Sampling ---
        loss = loss_fn(anchor_emb, positive_emb, negative_emb); loss.backward(); optimizer.step()
        # --- Validation Check (monitoring) ---
        if (epoch+1)%5==0 or epoch==config['epochs']-1: val_ndcg, val_f1 = evaluate_model(model, validation_papers, train_graph_data, all_papers_features, all_papers_graph_data, config, start_eval_paper_index=train_set_size); print(f"\nEpoch {epoch+1:02d}/{config['epochs']}, Loss: {loss.item():.4f}, Val NDCG: {val_ndcg:.4f}, Val F1: {val_f1:.4f}")
    print("--- Training Complete ---")

    # --- 3. Final Test Set Evaluation & Tracing ---
    print("\n--- Evaluating Final Model on Test Set ---")
    final_author_embeddings, final_paper_embeddings = None, None
    with torch.no_grad(): h_dict_final = model(all_papers_features, train_graph_data.edge_index, all_papers_graph_data.num_nodes); final_author_embeddings = h_dict_final['author']; final_paper_embeddings = h_dict_final['paper']
    # --- Trace First/Last Paper Final Embeddings ---
    first_paper_final_emb, last_paper_final_emb = None, None; first_paper_idx_final, last_paper_idx_final = -1, -1; first_paper_id, last_paper_id = None, None
    if all_papers_with_embeddings: # Check list not empty
        first_paper_id = all_papers_with_embeddings[0].get('id'); last_paper_id = all_papers_with_embeddings[-1].get('id')
        if all_papers_graph_data.node_maps.get('paper'): 
            paper_map_final=all_papers_graph_data.node_maps['paper']
            if first_paper_id in paper_map_final: 
                first_paper_idx_final=paper_map_final[first_paper_id]
                # Check index bounds before accessing embedding
                if first_paper_idx_final < final_paper_embeddings.shape[0]: first_paper_final_emb=final_paper_embeddings[first_paper_idx_final].cpu().numpy()
            if last_paper_id in paper_map_final: 
                last_paper_idx_final=paper_map_final[last_paper_id]
                if last_paper_idx_final < final_paper_embeddings.shape[0]: 
                    last_paper_final_emb=final_paper_embeddings[last_paper_idx_final].cpu().numpy()
        print(f"\n--- Tracing First Paper ID: {first_paper_id} (Final GNN Index: {first_paper_idx_final}) ---")
        if first_paper_final_emb is not None: print(f"Final GNN Output Embedding (first 10 dims):\n{first_paper_final_emb[:10]}...\nShape: {first_paper_final_emb.shape}\n")
        else: print("Final GNN embedding not found (check index/filtering).")
        print(f"\n--- Tracing Last Paper ID: {last_paper_id} (Final GNN Index: {last_paper_idx_final}) ---")
        if last_paper_final_emb is not None: print(f"Final GNN Output Embedding (first 10 dims):\n{last_paper_final_emb[:10]}...\nShape: {last_paper_final_emb.shape}\n")
        else: print("Final GNN embedding not found (check index/filtering).")
    else: print("Warning: No papers available for tracing final embeddings.")
    # --- End Trace ---

    # Evaluate model on test set
    test_ndcg, test_f1 = evaluate_model(model, test_papers, train_graph_data, all_papers_features, all_papers_graph_data, config, start_eval_paper_index=train_set_size + validation_set_size)

    # --- Recalculate Precision and Recall ---
    print("Calculating final Precision and Recall...")
    model.eval(); total_precision, total_recall, evaluated_count_pr = 0.0, 0.0, 0; inference_author_map = all_papers_graph_data.node_maps.get('author', {})
    if not inference_author_map: print("Warning: Author map missing for P/R calc.")

    # Use final embeddings obtained above for P/R calculation
    if final_author_embeddings is not None and final_paper_embeddings is not None and inference_author_map:
        for i in tqdm(range(len(test_papers)), desc="Final P/R Calc", ncols=80):
            test_paper=test_papers[i]; test_paper_index=train_set_size+validation_set_size+i; true_author_ids_all=set( str(aid) for aid in test_paper.get(config['AUTHOR_ID_KEY'],[]) ); true_author_ids=true_author_ids_all.intersection(inference_author_map.keys());
            if not true_author_ids: continue;
            if test_paper_index >= final_paper_embeddings.shape[0]: continue;
            test_paper_embedding = final_paper_embeddings[test_paper_index].unsqueeze(0);
            if final_author_embeddings.nelement()==0: continue;
            similarities = F.cosine_similarity(test_paper_embedding, final_author_embeddings); candidate_scores = []
            for aid, idx in inference_author_map.items():
                if idx < similarities.shape[0]: candidate_scores.append({'id': str(aid), 'score': similarities[idx].item()}) # Force str
            if not candidate_scores: continue;
            candidate_scores.sort(key=lambda x: x['score'], reverse=True);
            # Use robust metric function
            precision, recall, _ = calculate_precision_recall_f1(true_author_ids_all, candidate_scores, config['K_AT']);
            total_precision+=precision; total_recall+=recall; evaluated_count_pr+=1
    avg_precision = total_precision/evaluated_count_pr if evaluated_count_pr > 0 else 0.0; avg_recall = total_recall/evaluated_count_pr if evaluated_count_pr > 0 else 0.0

    # --- 4. Report Final Metrics ---
    print("\n" + "=" * 50); print("✅ Final Model Evaluation Results ✅"); print("=" * 50)
    print(f"  - Embedding Model: {EMBEDDING_TYPE.upper()} ({EMBEDDING_MODEL_NAME})"); print(f"  - Dataset Size:    {num_papers}"); print(f"  - Neg Sampling:    Top-{config['hard_negative_k']} Hard"); print("-" * 50)
    print(f"  - Avg. NDCG       @{config['K_AT']}: {test_ndcg:.4f}"); print(f"  - Avg. Precision  @{config['K_AT']}: {avg_precision:.4f}"); print(f"  - Avg. Recall     @{config['K_AT']}: {avg_recall:.4f}"); print(f"  - Avg. F1-Score   @{config['K_AT']}: {test_f1:.4f}"); print("=" * 50)


if __name__ == "__main__":
    # --- Best hyperparameters from YOUR HPO run ---
    # !!! UPDATE THESE WITH YOUR ACTUAL BEST PARAMETERS for the specific dataset size !!!
    # Example using 1000-paper SPECTER+HardNeg HPO results (NDCG 0.1767)
    FINAL_CONFIG = {
        'hidden_dim': 256,
        'output_dim': 128,
        'lr': 0.00038513658695024655,
        'epochs': 37,
        'weight_decay': 4.6518674145775124e-05,
        'dropout': 0.3835705175412137,
        'margin': 0.5689691185205238,
        'hard_negative_k': 10, # Crucial: Include from HPO params
        # --- Keep these consistent ---
        'K_AT': 10,
        'AUTHOR_ID_KEY': 'author_id_list',
        'EMBEDDING_DIM': EMBEDDING_DIM,
        'EMBEDDING_KEY': embedding_key # Uses global key based on EMBEDDING_TYPE
    }

    # --- Set final dataset size ---
    MAX_PAPERS_FOR_FINAL_MODEL = 100 # Set to 1000 or 10000 etc.
    TEST_SET_RATIO = 0.2
    VALIDATION_SET_RATIO = 0.1 # Should match HPO script
    CACHE_DIRECTORY = "./embedding_cache" # Directory to store embedding caches

    print("Starting final model training and evaluation pipeline...")
    os.makedirs(CACHE_DIRECTORY, exist_ok=True) # Ensure cache dir exists

    papers_list = open_file()
    if papers_list:
        papers_with_embeddings = filter_and_preprocess_data(papers_list, MAX_PAPERS_FOR_FINAL_MODEL,
                                                           model_type=EMBEDDING_TYPE, cache_dir=CACHE_DIRECTORY)
        if papers_with_embeddings:
            # Check if embedding key exists in the first paper after loading/processing
            if not papers_with_embeddings or embedding_key not in papers_with_embeddings[0] or papers_with_embeddings[0].get(embedding_key) is None:
                 print(f"Error: Embedding key '{embedding_key}' not found or is None in processed papers.")
            else:
                train_and_evaluate_final_model(papers_with_embeddings, FINAL_CONFIG, TEST_SET_RATIO, VALIDATION_SET_RATIO)
        else:
            print("No valid papers to process after filtering.")
    else:
        print("Failed to load data. Exiting.")
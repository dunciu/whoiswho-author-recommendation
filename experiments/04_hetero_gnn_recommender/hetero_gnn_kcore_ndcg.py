import numpy as np
import json
import warnings
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict, Counter
import os
import random
import sys
import math

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Scikit-learn
try:
    from sklearn.mixture import GaussianMixture
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Warning: scikit-learn not found. Clustering strategies will be disabled.")

# Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
# ✅ UPDATED: Sizes to 1000/5000 and K to 5/10/20
SIZES_TO_TEST = [1000, 5000] 
K_VALUES_TO_TEST = [5, 10, 20]      
MASKS_TO_TEST = [1, 2, 3]           

# Data Filtering
MIN_AUTHORS = 3
K_CORE_MIN = 4              
MAX_AUTHORS_PER_PAPER = 10 

# Model Hyperparameters
EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2' 
embedding_key = 'mpnet_embedding'
HIDDEN_DIM = 256        
EPOCHS = 100            
LR = 0.0005             
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ---------------------------------------------------------
# 2. DATA LOADING & FILTERING
# ---------------------------------------------------------
embed_model = None

def load_embedding_model():
    global embed_model
    if not SENTENCE_TRANSFORMER_AVAILABLE:
        print("❌ CRITICAL: sentence-transformers not installed.")
        sys.exit(1)
    if embed_model is None:
        try:
            print(f"⏳ Loading AI Brain ({EMBEDDING_MODEL_NAME})...")
            embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=str(DEVICE))
            print(f"✅ Model loaded on {DEVICE}!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)

def get_embedding(text_list):
    return embed_model.encode(text_list, convert_to_numpy=True, show_progress_bar=False)

def extract_author_ids(paper):
    ids = []
    if 'author_id_list' in paper and paper['author_id_list']:
        return paper['author_id_list']
    for a in paper.get('authors', []):
        if isinstance(a, dict) and 'name' in a:
            ids.append(str(a['name']).strip())
    return list(set(ids))

def generate_synthetic_data(n):
    """Generates clean synthetic papers (Fallback logic)."""
    print(f"   ⚡ Generating {n} synthetic papers (fallback)...")
    papers = []
    n_groups = max(5, n // 50) 
    groups = []
    for g in range(n_groups):
        topic = f"Topic_{g}"
        authors = [f"Res_{g}_{i}" for i in range(15)]
        groups.append((topic, authors))

    for i in range(n):
        topic, group_authors = random.choice(groups)
        p_auths = random.sample(group_authors, random.randint(3, 5))
        papers.append({
            'id': f"P{i}", 
            'title': f"{topic} Investigation {i}",
            'abstract': f"This is a study about {topic}.",
            'authors': [{'name': a} for a in p_auths]
        })
    return papers

def get_k_core_subset(target_size, k=K_CORE_MIN):
    filename = "data/pid_to_info_all.json"
    data = []
    
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                raw_data = list(json.load(f).values())
            candidates = []
            for p in raw_data:
                auths = extract_author_ids(p)
                if MIN_AUTHORS <= len(auths) <= MAX_AUTHORS_PER_PAPER:
                    candidates.append(p)
            data = candidates
        except: pass
    
    if len(data) > 0:
        print(f"   🔍 Applying Recursive K-Core Filter (k={k})...")
        current_papers = data
        for i in range(10):
            auth_counts = Counter()
            for p in current_papers:
                for a in extract_author_ids(p): auth_counts[a] += 1
            
            next_papers = []
            for p in current_papers:
                auths = extract_author_ids(p)
                strong_auths = [a for a in auths if auth_counts[a] >= k]
                if len(strong_auths) >= MIN_AUTHORS:
                    p['clean_authors'] = strong_auths 
                    next_papers.append(p)
            if len(next_papers) == len(current_papers): break 
            current_papers = next_papers
        data = current_papers
        print(f"   ✅ Graph stabilized with {len(data)} papers.")

    if len(data) < target_size:
        needed = target_size - len(data)
        data.extend(generate_synthetic_data(needed))
    
    if len(data) > target_size:
        data = random.sample(data, target_size)

    # Embedding
    valid_papers = []
    texts_to_embed = []
    indices_to_embed = []

    for i, p in enumerate(data):
        if 'clean_authors' not in p: p['clean_authors'] = extract_author_ids(p)
        if len(p['clean_authors']) >= MIN_AUTHORS:
            if embedding_key not in p:
                text = f"{p.get('title','')} [SEP] {p.get('abstract','')}"
                texts_to_embed.append(text)
                indices_to_embed.append(i)
            valid_papers.append(p)
            
    if texts_to_embed:
        print(f"   🧠 Embedding {len(texts_to_embed)} papers...")
        batch_size = 32
        embs = []
        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[i:i+batch_size]
            embs.append(get_embedding(batch_texts))
        
        full_emb = np.concatenate(embs)
        for idx, vec in zip(indices_to_embed, full_emb):
            data[idx][embedding_key] = vec
                
    return [p for p in valid_papers if embedding_key in p]

# ---------------------------------------------------------
# 3. GNN MODEL
# ---------------------------------------------------------
class HeteroGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM): 
        super().__init__()
        self.proj_p = nn.Linear(input_dim, hidden_dim)
        self.proj_a = nn.Linear(input_dim, hidden_dim)
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, p_feats, a_feats, edge_index):
        h_p = F.relu(self.proj_p(p_feats))
        h_a = F.relu(self.proj_a(a_feats))
        
        src, dst = edge_index 
        
        msg = torch.zeros_like(h_p)
        msg.index_add_(0, dst, h_a[src])
        
        deg = torch.bincount(dst, minlength=h_p.size(0)).float().unsqueeze(1).clamp(min=1)
        h_p_upd = h_p + self.dropout(F.relu(self.W(msg / deg)))
        
        return h_p_upd, h_a

def train_model(papers, author_map):
    dim = len(papers[0][embedding_key])
    p_feats = torch.tensor(np.stack([p[embedding_key] for p in papers]), dtype=torch.float32).to(DEVICE)
    
    a_init = torch.zeros((len(author_map), dim), device=DEVICE)
    counts = torch.zeros((len(author_map), 1), device=DEVICE)
    
    src, dst = [], []
    for i, p in enumerate(papers):
        p_vec = torch.tensor(p[embedding_key], device=DEVICE)
        for a in p['clean_authors']:
            if a in author_map:
                idx = author_map[a]
                a_init[idx] += p_vec
                counts[idx] += 1
                src.append(idx); dst.append(i)
                
    a_init = a_init / counts.clamp(min=1)
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=DEVICE)
    
    model = HeteroGNN(dim, hidden_dim=HIDDEN_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    
    model.train()
    for _ in range(EPOCHS):
        h_p, h_a = model(p_feats, a_init, edge_index)
        anc = h_p[dst]
        pos = h_a[src]
        neg = pos[torch.randperm(pos.size(0))] 
        loss = F.triplet_margin_loss(anc, pos, neg, margin=0.5)
        opt.zero_grad(); loss.backward(); opt.step()
        
    model.eval()
    with torch.no_grad(): _, final_a_embs = model(p_feats, a_init, edge_index)
    return model, final_a_embs

# ---------------------------------------------------------
# 4. RECOMMENDER
# ---------------------------------------------------------
class Recommender:
    def __init__(self, model, a_embs, a_map):
        self.model = model
        self.a_embs = a_embs
        self.a_map = a_map
        self.idx2a = {v: k for k, v in a_map.items()}
        self.clusters = defaultdict(list)
        self.gmm = None
        
    def fit_clusters(self, n_clusters=12):
        if SKLEARN_AVAILABLE:
            self.gmm = GaussianMixture(n_components=n_clusters, random_state=SEED)
            emb_norm = F.normalize(self.a_embs, p=2, dim=1).cpu().numpy()
            self.gmm.fit(emb_norm)
            preds = self.gmm.predict(emb_norm)
            for i, c in enumerate(preds): self.clusters[c].append(i)

    def predict(self, p_emb, known_author_ids=None, k=20, strategy='none'):
        """
        Supports k=20 max for slicing later.
        """
        with torch.no_grad():
            p_vec = torch.tensor(p_emb, device=DEVICE).unsqueeze(0)
            h_p = F.relu(self.model.proj_p(p_vec))
            
            if known_author_ids:
                known_indices = [self.a_map[a] for a in known_author_ids if a in self.a_map]
                if known_indices:
                    hint_feats = self.a_embs[known_indices] 
                    team_msg = torch.mean(hint_feats, dim=0, keepdim=True)
                    h_p = h_p + self.model.dropout(F.relu(self.model.W(team_msg)))

            p_proj = F.normalize(h_p, p=2, dim=1)
            a_norm = F.normalize(self.a_embs, p=2, dim=1)
            scores = torch.mm(p_proj, a_norm.t()).squeeze(0)
            
            if known_author_ids:
                for a in known_author_ids:
                    if a in self.a_map:
                        scores[self.a_map[a]] = -999.0

            if strategy != 'none' and self.gmm:
                probs = self.gmm.predict_proba(p_proj.cpu().numpy())[0]
                if strategy == 'hard':
                    top_c = np.argmax(probs)
                    mask = torch.ones_like(scores) * -999.0
                    if self.clusters[top_c]: mask[self.clusters[top_c]] = 0.0
                    scores += mask
                elif strategy == 'soft':
                    top_3 = np.argsort(probs)[-3:]
                    for c in top_3:
                        bonus = 0.4 * probs[c] 
                        scores[self.clusters[c]] += bonus

            vals, inds = torch.topk(scores, min(k, len(self.a_map)))
            return [(self.idx2a[i.item()], v.item()) for i, v in zip(inds, vals)]

# ---------------------------------------------------------
# 5. METRICS (Acc, Prec, Rec, NDCG)
# ---------------------------------------------------------
# ✅ NEW FUNCTION: NDCG Calculation
def calculate_ndcg(ranked_list, true_set, k):
    """
    Normalized Discounted Cumulative Gain at K.
    Binary relevance: 1 if in true_set, 0 otherwise.
    """
    dcg = 0.0
    idcg = 0.0
    
    # 1. Calculate DCG for our results
    for i, (name, score) in enumerate(ranked_list[:k]):
        if name in true_set:
            rel = 1.0
            dcg += rel / math.log2(i + 2) # i+2 because rank starts at 1, so log2(rank+1)

    # 2. Calculate IDCG (Ideal case: all True items are at the top)
    num_relevant = min(len(true_set), k)
    for i in range(num_relevant):
        rel = 1.0 # In ideal case, we find a relevant item
        idcg += rel / math.log2(i + 2)
        
    if idcg == 0.0: return 0.0
    return dcg / idcg

def calculate_metrics(preds, hidden_truth, k):
    # Slice predictions to top-k
    top_k_preds = preds[:k]
    pred_names = [p[0] for p in top_k_preds]
    
    hits = len(set(pred_names).intersection(set(hidden_truth)))
    
    # Metrics
    acc = 1.0 if hits > 0 else 0.0
    prec = hits / k
    rec = hits / len(hidden_truth) if hidden_truth else 0.0
    ndcg = calculate_ndcg(preds, set(hidden_truth), k)
    
    return acc, prec, rec, ndcg

# ---------------------------------------------------------
# 6. MAIN EXPERIMENT
# ---------------------------------------------------------
def run_experiment():
    load_embedding_model()
    
    print(f"\n STARTING EXTENDED EXPERIMENT")
    print(f"   Sizes: {SIZES_TO_TEST}")
    print(f"   Cutoffs (K): {K_VALUES_TO_TEST}")
    print(f"   Strategies: None, Hard, Soft (Comparing Clustering)")
    
    for size in SIZES_TO_TEST:
        print(f"\n" + "="*70)
        print(f"DATASET SIZE: {size}")
        print("="*70)
        
        # 1. Load & Split
        papers = get_k_core_subset(size, k=K_CORE_MIN)
        if len(papers) < 50: 
            print("Not enough papers after filtering.")
            continue
        
        train_p, test_p = train_test_split(papers, test_size=0.2, random_state=SEED)
        
        # 2. Train
        auth_set = set()
        for p in train_p: auth_set.update(p['clean_authors'])
        author_map = {n: i for i, n in enumerate(auth_set)}
        
        print(f"   Training on {len(train_p)} papers, {len(author_map)} authors...")
        model, a_vecs = train_model(train_p, author_map)
        rec = Recommender(model, a_vecs, author_map)
        rec.fit_clusters()
        
        # 3. Evaluate loop
        # Header
        print(f"\n   {'Mask':<4} | {'Strat':<6} | {'K':<3} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'NDCG':<6}")
        print("-" * 60)
        
        for mask_n in MASKS_TO_TEST:
            for strat in ['none', 'soft']: # Focusing on None vs Soft
                # UPDATED: Loop through K values
                for k_val in K_VALUES_TO_TEST:
                    
                    metrics = {'acc': [], 'prec': [], 'rec': [], 'ndcg': []}
                    
                    for p in test_p:
                        poss = [a for a in p['clean_authors'] if a in author_map]
                        if len(poss) <= mask_n: continue 
                        
                        poss.sort()
                        random.Random(p['id']).shuffle(poss)
                        
                        hidden = poss[:mask_n]    # Targets
                        known = poss[mask_n:]     # Hints
                        
                        # Get max necessary predictions (20) then slice in metrics function
                        preds = rec.predict(p[embedding_key], known_author_ids=known, k=20, strategy=strat)
                        
                        a, pr, re, nd = calculate_metrics(preds, hidden, k_val)
                        metrics['acc'].append(a)
                        metrics['prec'].append(pr)
                        metrics['rec'].append(re)
                        metrics['ndcg'].append(nd)
                    
                    if metrics['acc']:
                        print(f"   {mask_n:<4} | {strat:<6} | {k_val:<3} | "
                              f"{np.mean(metrics['acc']):.4f} | "
                              f"{np.mean(metrics['prec']):.4f} | "
                              f"{np.mean(metrics['rec']):.4f} | "
                              f"{np.mean(metrics['ndcg']):.4f}")

    print("\n Experiment Complete.")

if __name__ == "__main__":
    run_experiment()
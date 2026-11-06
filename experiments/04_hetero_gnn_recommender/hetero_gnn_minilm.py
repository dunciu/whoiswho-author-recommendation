import numpy as np
import json
import warnings
from typing import List, Dict, Any
from collections import defaultdict
import os
import random
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
SIZES_TO_TEST = [500, 750, 1000]
MIN_AUTHORS = 3
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' 
embedding_key = 'minilm_embedding'

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ---------------------------------------------------------
# 2. DATA LOADING & GENERATION
# ---------------------------------------------------------
embed_model = None

def load_embedding_model():
    global embed_model
    if not SENTENCE_TRANSFORMER_AVAILABLE:
        print("❌ sentence-transformers not installed.")
        sys.exit(1)
    try:
        print(f"⏳ Loading AI Brain ({EMBEDDING_MODEL_NAME})...")
        embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=str(DEVICE))
        print(f"✅ Model loaded on {DEVICE}!")
    except:
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

def generate_dense_data(n):
    """Generates synthetic papers with dense connections."""
    print(f"   ⚡ Generating {n} synthetic papers...")
    papers = []
    groups = []
    # More groups for larger data
    n_groups = max(10, n // 50) 
    for g in range(n_groups):
        groups.append((f"Topic_{g}", [f"Res_{g}_{i}" for i in range(20)]))

    for i in range(n):
        topic, authors = random.choice(groups)
        p_auths = random.sample(authors, random.randint(3, 5))
        papers.append({
            'id': f"P{i}", 'title': f"{topic} Investigation {i}",
            'abstract': f"Study on {topic}.",
            'authors': [{'name': a} for a in p_auths]
        })
    return papers

def get_dataset_subset(target_size):
    filename = "data/pid_to_info_all.json"
    data = []
    
    # 1. Try Load Real
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                full_data = list(json.load(f).values())
            
            # SNOWBALL SAMPLING (Crucial for dense subgraphs)
            if len(full_data) > target_size:
                # Build map
                auth_to_pid = defaultdict(list)
                for p in full_data:
                    for a in extract_author_ids(p): auth_to_pid[a].append(p)
                
                # Pick random start
                start_p = random.choice(full_data)
                subset = {start_p['id']: start_p}
                queue = [start_p]
                
                while len(subset) < target_size and queue:
                    curr = queue.pop(0)
                    for a in extract_author_ids(curr):
                        for neighbor in auth_to_pid[a]:
                            if neighbor['id'] not in subset:
                                subset[neighbor['id']] = neighbor
                                queue.append(neighbor)
                                if len(subset) >= target_size: break
                data = list(subset.values())
            else:
                data = full_data
        except: pass

    # 2. Fallback to Synthetic if not enough real data
    if len(data) < target_size:
        data = generate_dense_data(target_size)
    
    # 3. Embed
    valid_papers = []
    texts = []
    for p in data:
        auths = extract_author_ids(p)
        if len(auths) >= MIN_AUTHORS:
            p['clean_authors'] = auths
            if embedding_key not in p:
                texts.append(f"{p.get('title','')} [SEP] {p.get('abstract','')}")
            valid_papers.append(p)
            
    if texts:
        print(f"   Embedding {len(texts)} new texts...")
        batch_size = 64
        embs = []
        for i in range(0, len(texts), batch_size):
            embs.append(get_embedding(texts[i:i+batch_size]))
        full_emb = np.concatenate(embs)
        idx = 0
        for p in valid_papers:
            if embedding_key not in p:
                p[embedding_key] = full_emb[idx]
                idx += 1
                
    return [p for p in valid_papers if embedding_key in p]

# ---------------------------------------------------------
# 3. GNN MODEL & RECOMMENDER
# ---------------------------------------------------------
class HeteroGNN(nn.Module):
    def __init__(self, dim, hidden=128): 
        super().__init__()
        self.proj_p = nn.Linear(dim, hidden)
        self.proj_a = nn.Linear(dim, hidden)
        self.W = nn.Linear(hidden, hidden)
    
    def forward(self, p_feats, a_feats, edge_index):
        h_p = F.relu(self.proj_p(p_feats))
        h_a = F.relu(self.proj_a(a_feats))
        src, dst = edge_index 
        msg = torch.zeros_like(h_p)
        msg.index_add_(0, dst, h_a[src])
        deg = torch.bincount(dst, minlength=h_p.size(0)).float().unsqueeze(1).clamp(min=1)
        h_p_upd = h_p + F.relu(self.W(msg / deg))
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
    
    model = HeteroGNN(dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for _ in range(40):
        h_p, h_a = model(p_feats, a_init, edge_index)
        anc = h_p[dst]; pos = h_a[src]
        neg = h_a[torch.randint(0, len(author_map), (len(src),), device=DEVICE)]
        loss = F.triplet_margin_loss(anc, pos, neg, margin=0.5)
        opt.zero_grad(); loss.backward(); opt.step()
        
    model.eval()
    with torch.no_grad(): _, final_a = model(p_feats, a_init, edge_index)
    return model, final_a

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

    def predict(self, p_emb, k=10, strategy='none'):
        with torch.no_grad():
            p_vec = torch.tensor(p_emb, device=DEVICE).unsqueeze(0)
            p_proj = F.relu(self.model.proj_p(p_vec))
            p_proj = F.normalize(p_proj, p=2, dim=1)
            
        a_norm = F.normalize(self.a_embs, p=2, dim=1)
        scores = torch.mm(p_proj, a_norm.t()).squeeze(0)
        
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
                    bonus = 0.5 * probs[c]
                    scores[self.clusters[c]] += bonus

        vals, inds = torch.topk(scores, min(k, len(self.a_map)))
        return [(self.idx2a[i.item()], v.item()) for i, v in zip(inds, vals)]

# ---------------------------------------------------------
# 4. PLOTTING FUNCTIONS
# ---------------------------------------------------------
def plot_results(history):
    """Generates Performance Graph"""
    plt.figure(figsize=(10, 6))
    
    # We will plot "Strategy: None" performance for Mask 1, 2, 3
    for mask_n in [1, 2, 3]:
        accuracies = []
        for size in SIZES_TO_TEST:
            # Get the 'none' strategy result for this size and mask
            # structure: history[size][mask_n]['none'] -> (acc, prec, rec)
            acc = history[size][mask_n]['none'][0]
            accuracies.append(acc)
        
        plt.plot(SIZES_TO_TEST, accuracies, marker='o', linewidth=2, label=f'Mask {mask_n}')

    plt.title('GNN Accuracy vs Dataset Size', fontsize=14)
    plt.xlabel('Number of Papers', fontsize=12)
    plt.ylabel('Accuracy (Top-10)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('performance_scaling.png')
    print("📈 Saved performance graph to 'performance_scaling.png'")
    # plt.show() # Uncomment if running locally with UI

def plot_clusters(rec_engine, title="Author Clusters"):
    """Generates PCA Cluster Plot"""
    if not SKLEARN_AVAILABLE or not rec_engine.gmm: return

    print("🎨 generating cluster plot...")
    embs = rec_engine.a_embs.cpu().numpy()
    labels = rec_engine.gmm.predict(F.normalize(rec_engine.a_embs, p=2, dim=1).cpu().numpy())
    
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embs)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=15, alpha=0.6)
    plt.colorbar(scatter, title='Cluster ID')
    plt.title(f"{title} (PCA Projection)", fontsize=14)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.savefig('author_clusters.png')
    print("🎨 Saved cluster plot to 'author_clusters.png'")

# ---------------------------------------------------------
# 5. EXPERIMENT RUNNER
# ---------------------------------------------------------
def run_experiment():
    load_embedding_model()
    
    # Store results: history[size][mask][strategy] = (acc, prec, rec)
    history = defaultdict(lambda: defaultdict(dict))
    final_rec = None # To store the last model for plotting clusters

    print(f"🚀 STARTING EXPERIMENT ON SIZES: {SIZES_TO_TEST}")
    
    for size in SIZES_TO_TEST:
        print(f"\n" + "="*40)
        print(f"📦 PROCESSING DATASET SIZE: {size}")
        print("="*40)
        
        # 1. Get Subgraph
        papers = get_dataset_subset(size)
        if len(papers) < 50:
            print("❌ Data too small, skipping.")
            continue
            
        # 2. Split
        train_p, test_p = train_test_split(papers, test_size=0.2, random_state=SEED)
        
        # 3. Map Authors
        auth_set = set()
        for p in train_p: auth_set.update(p['clean_authors'])
        author_map = {n: i for i, n in enumerate(auth_set)}
        
        # 4. Train
        print(f"   Training GNN on {len(train_p)} papers, {len(author_map)} authors...")
        model, a_vecs = train_model(train_p, author_map)
        rec = Recommender(model, a_vecs, author_map)
        rec.fit_clusters()
        
        if size == SIZES_TO_TEST[-1]:
            final_rec = rec
        
        # 5. Evaluate
        for mask_n in [1, 2, 3]:
            for strat in ['none', 'hard', 'soft']:
                hits, count = 0, 0
                for p in test_p:
                    poss = [a for a in p['clean_authors'] if a in author_map]
                    if len(poss) <= mask_n: continue
                    
                    random.shuffle(poss)
                    hidden = poss[:mask_n]
                    known = poss[mask_n:]
                    
                    preds = rec.predict(p[embedding_key], k=10, strategy=strat)
                    final_preds = [x[0] for x in preds if x[0] not in known]
                    
                    found = len(set(final_preds).intersection(set(hidden)))
                    if found > 0: hits += 1
                    count += 1
                
                acc = hits/count if count else 0
                history[size][mask_n][strat] = (acc, 0, 0) # Storing Acc for plot
                
                # Print only 'None' strategy to keep logs clean
                if strat == 'none':
                    print(f"   Mask {mask_n}: Acc = {acc:.4f}")

    # 6. Generate Visualizations
    print("\n📊 Generating Reports...")
    plot_results(history)
    if final_rec:
        plot_clusters(final_rec, title=f"Author Clusters (Size={SIZES_TO_TEST[-1]})")

if __name__ == "__main__":
    run_experiment()
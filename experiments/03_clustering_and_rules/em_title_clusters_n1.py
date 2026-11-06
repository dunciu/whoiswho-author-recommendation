import numpy as np
import time
import json
import math
import warnings
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
import itertools
import os
import shutil
import string
import pickle
import random
from tqdm import tqdm

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# 1. IMPORTS & DEPENDENCY CHECKS
# ---------------------------------------------------------

# SENTENCE TRANSFORMERS (For SPECTER - High Score)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    print("Warning: sentence-transformers not found. Install it via: pip install sentence-transformers")

# OPTUNA (For HPO)
import optuna

# SCIKIT-LEARN (For EM Clustering & Hard Negatives)
try:
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not found. Install it via: pip install scikit-learn")

# MLXTEND (For Association Rules)
try:
    import pandas as pd
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False
    print("Warning: mlxtend not found. Install via: pip install mlxtend pandas")

# Suppress warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 2. CONFIGURATION & SEEDING
# ---------------------------------------------------------
SEED = 42
def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")

set_seed(SEED)

EMBEDDING_MODEL_NAME = 'allenai/specter'
EMBEDDING_TYPE = 'specter'
EMBEDDING_DIM = 768
embedding_key = f'{EMBEDDING_TYPE}_embedding'

# ---------------------------------------------------------
# 3. EMBEDDING MODEL (SPECTER)
# ---------------------------------------------------------
def initialize_embedding_model(model_name: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = None
    if SENTENCE_TRANSFORMER_AVAILABLE:
        try:
            model = SentenceTransformer(model_name, device=str(device))
            print(f"SPECTER Model loaded on: {device}")
        except Exception as e:
            print(f"Error loading SPECTER: {e}")
    return model, device

embed_model, DEVICE = initialize_embedding_model(EMBEDDING_MODEL_NAME)

def get_embedding(text_list: List[str]) -> np.ndarray:
    """Generates SPECTER embeddings."""
    if not text_list or embed_model is None:
        return np.random.rand(len(text_list), EMBEDDING_DIM).astype(np.float32)
    try:
        # SPECTER handles batching internally
        embeddings = embed_model.encode(text_list, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.astype(np.float32)
    except Exception as e:
        print(f"Embedding error: {e}")
        return np.random.rand(len(text_list), EMBEDDING_DIM).astype(np.float32)

# ---------------------------------------------------------
# 4. DATA LOADING & PREPROCESSING
# ---------------------------------------------------------
def extract_paper_author_ids(paper_data: Dict[str, Any]) -> List[str]:
    """Extracts author IDs (Names) from paper data."""
    if 'author_id_list' in paper_data and paper_data['author_id_list']:
        return paper_data['author_id_list']
    
    authors = paper_data.get('authors', [])
    if not authors or not isinstance(authors, list): return []
    
    author_ids = []
    # Using 'name' as ID since dataset lacks numeric IDs
    for author in authors:
        if not isinstance(author, dict): continue
        if 'name' in author and author['name']:
            val = str(author['name']).strip()
            if val: author_ids.append(val)
            
    return list(set(author_ids))

def preprocess_papers_with_embeddings(papers: List[Dict[str, Any]], cache_path: str) -> List[Dict[str, Any]]:
    """Generates SPECTER embeddings (Title + Abstract)."""
    cached_embeddings = {}
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}...")
        try:
            with open(cache_path, 'rb') as f: cached_embeddings = pickle.load(f)
            print(f"Loaded {len(cached_embeddings)} embeddings.")
        except: cached_embeddings = {}

    papers_req = []
    for paper in papers:
        pid = paper.get('id')
        if pid and pid in cached_embeddings:
            paper[embedding_key] = cached_embeddings[pid]
        elif embedding_key not in paper:
            if pid: papers_req.append(paper)

    if not papers_req:
        print("All embeddings ready.")
        return [p for p in papers if p.get(embedding_key) is not None]

    print(f"Generating SPECTER embeddings for {len(papers_req)} papers...")
    texts_to_embed = []
    for p in papers_req:
        # SPECTER INPUT FORMAT: Title + [SEP] + Abstract
        title = p.get('title', '')
        abstract = p.get('abstract', '') or ''
        combined = f"{title} {embed_model.tokenizer.sep_token} {abstract}"
        texts_to_embed.append(combined)

    all_gen = []
    batch_size = 32
    for i in tqdm(range(0, len(texts_to_embed), batch_size), desc="Embedding"):
        batch = texts_to_embed[i:i+batch_size]
        all_gen.append(get_embedding(batch))
    
    if all_gen:
        final_emb = np.concatenate(all_gen, axis=0)
        new_data = {}
        for i, paper in enumerate(papers_req):
            paper[embedding_key] = final_emb[i]
            new_data[paper['id']] = final_emb[i]
        
        cached_embeddings.update(new_data)
        with open(cache_path, 'wb') as f: pickle.dump(cached_embeddings, f)

    return [p for p in papers if p.get(embedding_key) is not None]

def open_file(filename: str = "data/pid_to_info_all.json") -> List[Dict[str, Any]]:
    try:
        with open(filename, 'r', encoding='utf-8') as f: 
            data = json.load(f)
            return list(data.values())
    except:
        print("File not found. Generating dense dummy data.")
        return generate_dummy_data(500)

def generate_dummy_data(num_papers: int) -> List[Dict[str, Any]]:
    papers = []
    num_authors = max(10, num_papers // 5) 
    authors = [f"Author_{i:03d}" for i in range(num_authors)]
    topics = ["Deep Learning", "Databases", "Robotics", "HCI", "Theory"]
    for i in range(num_papers):
        t = topics[i % 5]
        p_auths = np.random.choice(authors, np.random.randint(2, 5), replace=False).tolist()
        papers.append({
            'id': f"P{i}", 'title': f"{t} Paper {i}", 'abstract': f"About {t}...",
            'authors': [{'name': a, 'org': 'University of Dummy'} for a in p_auths], 'year': 2023
        })
    return papers

def filter_and_preprocess_data(papers, max_p, cache_dir):
    valid = []
    print(f"Filtering {len(papers)} papers...")
    for p in papers:
        a_ids = extract_paper_author_ids(p)
        if (p.get('title') or p.get('abstract')) and a_ids:
            p['author_id_list'] = a_ids
            valid.append(p)
    
    if max_p and len(valid) > max_p:
        # Sort by year if possible
        try:
            valid.sort(key=lambda x: int(x.get('year', 0) or 0), reverse=True)
        except: pass
        valid = valid[:max_p]
        
    print(f"Processing {len(valid)} valid papers.")
    path = os.path.join(cache_dir, f"specter_embeddings_{len(valid)}.pkl")
    return preprocess_papers_with_embeddings(valid, path)

# ---------------------------------------------------------
# 5. GRAPH & GNN
# ---------------------------------------------------------
class ManualGraphData:
    def __init__(self): self.node_features={}; self.edge_index={}; self.node_maps={}; self.num_nodes={}

class ManualHeteroGNN(nn.Module):
    def __init__(self, pi, h, o, d): 
        super().__init__()
        self.lp=nn.Linear(pi,h); self.la=nn.Linear(pi,h); self.drop=nn.Dropout(d)
        self.w_ap=nn.Linear(h,h); self.w_pa=nn.Linear(h,h); self.w_co=nn.Linear(h,h)
        self.olp=nn.Linear(h,o); self.ola=nn.Linear(h,o)
    
    def gcn(self, x_src, x_dst, idx, lin, n_dst):
        s, d = idx
        msg = torch.zeros(n_dst, x_src.size(1), device=x_src.device)
        msg.index_add_(0, d, x_src[s])
        deg = torch.bincount(d, minlength=n_dst).float().unsqueeze(1).clamp(min=1)
        return F.relu(lin(self.drop(msg/deg)))

    def forward(self, x, e, n):
        hp = F.relu(self.lp(x['paper'])); ha = F.relu(self.la(x['author']))
        mp, ma = torch.zeros_like(hp), torch.zeros_like(ha)
        if ('a','w','p') in e: mp += self.gcn(ha, hp, e[('a','w','p')], self.w_ap, n['paper'])
        if ('p','wb','a') in e: ma += self.gcn(hp, ha, e[('p','wb','a')], self.w_pa, n['author'])
        if ('a','co','a') in e: ma += self.gcn(ha, ha, e[('a','co','a')], self.w_co, n['author'])
        return {'paper': self.olp(hp+mp), 'author': self.ola(ha+ma)}

def build_graph(papers, auth_key, emb_key):
    aids = set(aid for p in papers for aid in p.get(auth_key, []))
    if not aids: return ManualGraphData()
    
    a_map = {aid: i for i, aid in enumerate(aids)}
    p_map = {p['id']: i for i, p in enumerate(papers)}
    
    a_embs = defaultdict(list)
    for p in papers:
        for aid in p.get(auth_key, []):
            if aid in a_map: a_embs[aid].append(p[emb_key])
            
    a_feat = np.zeros((len(a_map), EMBEDDING_DIM), dtype=np.float32)
    for aid, embs in a_embs.items(): a_feat[a_map[aid]] = np.mean(embs, axis=0)
    
    p_feat = torch.tensor(np.stack([p[emb_key] for p in papers]), dtype=torch.float32)
    
    g = ManualGraphData()
    g.node_features = {'paper': p_feat, 'author': torch.tensor(a_feat, dtype=torch.float32)}
    g.num_nodes = {'paper': len(papers), 'author': len(a_map)}
    g.node_maps = {'paper': p_map, 'author': a_map}
    
    asrc, adst, co = [], [], set()
    for p in papers:
        pidx = p_map[p['id']]
        aidxs = [a_map[aid] for aid in p.get(auth_key, []) if aid in a_map]
        for idx in aidxs: asrc.append(idx); adst.append(pidx)
        for a1, a2 in itertools.combinations(aidxs, 2): co.add(tuple(sorted((a1, a2))))
        
    if asrc:
        g.edge_index[('a','w','p')] = torch.tensor([asrc, adst], dtype=torch.long)
        g.edge_index[('p','wb','a')] = torch.tensor([adst, asrc], dtype=torch.long)
    
    c_s, c_d = [], []
    for a1, a2 in co: c_s.extend([a1, a2]); c_d.extend([a2, a1])
    if c_s: g.edge_index[('a','co','a')] = torch.tensor([c_s, c_d], dtype=torch.long)
    
    return g

# ---------------------------------------------------------
# 6. HPO OBJECTIVE (Hard Negative Sampling)
# ---------------------------------------------------------
def evaluate(model, papers, g, all_feats, n_nodes, k):
    model.eval()
    with torch.no_grad():
        out = model(all_feats, g.edge_index, n_nodes)
        p_emb, a_emb = out['paper'], out['author']
    
    amap = g.node_maps['author']
    total_ndcg, count = 0.0, 0
    
    for i, p in enumerate(papers):
        true_aids = set(str(a) for a in p.get('author_id_list', []))
        if not true_aids.intersection(amap.keys()): continue
        if p['id'] not in g.node_maps['paper']: continue
        pidx = g.node_maps['paper'][p['id']]
        
        p_vec = p_emb[pidx].unsqueeze(0)
        sims = F.cosine_similarity(p_vec, a_emb)
        
        cands = [{'id': aid, 'score': sims[uidx].item()} for aid, uidx in amap.items()]
        cands.sort(key=lambda x: x['score'], reverse=True)
        
        rel = [1 if c['id'] in true_aids else 0 for c in cands[:k]]
        dcg = sum((r / math.log2(i + 2)) for i, r in enumerate(rel))
        idcg = sum((1 / math.log2(i + 2)) for i in range(min(len(true_aids), k)))
        if idcg > 0: total_ndcg += dcg / idcg
        count += 1
         
    return total_ndcg / count if count > 0 else 0.0

def objective(trial, all_papers):
    cfg = {
        'h': trial.suggest_categorical('h', [128, 256]),
        'o': trial.suggest_categorical('o', [64, 128]),
        'lr': trial.suggest_float('lr', 1e-4, 1e-3, log=True),
        'ep': trial.suggest_int('ep', 20, 30),
        'wd': trial.suggest_float('wd', 1e-5, 1e-3, log=True),
        'dr': trial.suggest_float('dr', 0.2, 0.5),
        'mg': trial.suggest_float('mg', 0.4, 0.8)
    }
    
    n = len(all_papers)
    if n < 20: raise optuna.exceptions.TrialPruned()
    n_tr = int(n*0.6); n_val = int(n*0.2)
    tr_p = all_papers[:n_tr]; val_p = all_papers[n_tr:n_tr+n_val]
    
    full_g = build_graph(all_papers, 'author_id_list', embedding_key)
    if not full_g.edge_index: raise optuna.exceptions.TrialPruned()
    
    for k in full_g.node_features: full_g.node_features[k] = full_g.node_features[k].to(DEVICE)
    for k in full_g.edge_index: full_g.edge_index[k] = full_g.edge_index[k].to(DEVICE)
    
    tr_edges = full_g.edge_index[('a','w','p')]
    mask = tr_edges[1] < n_tr
    tr_edges = tr_edges[:, mask]
    
    use_hard = False
    clusters = defaultdict(list); labels = None
    if SKLEARN_AVAILABLE:
        try:
            feats = full_g.node_features['author'].cpu().numpy()
            if len(feats) > 20:
                kmeans = KMeans(n_clusters=min(50, len(feats)//3), random_state=SEED, n_init=3).fit(feats)
                labels = torch.tensor(kmeans.labels_, device=DEVICE)
                for i, l in enumerate(kmeans.labels_): clusters[l].append(i)
                use_hard = True
        except: pass

    model = ManualHeteroGNN(EMBEDDING_DIM, cfg['h'], cfg['o'], cfg['dr']).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    loss_f = nn.TripletMarginLoss(margin=cfg['mg'])
    
    n_auth = full_g.num_nodes['author']
    
    for ep in range(cfg['ep']):
        model.train()
        opt.zero_grad()
        out = model(full_g.node_features, full_g.edge_index, full_g.num_nodes)
        hp, ha = out['paper'], out['author']
        
        anc = hp[tr_edges[1]]; pos = ha[tr_edges[0]]
        
        if use_hard:
            p_idx = tr_edges[0]
            p_cls = labels[p_idx]
            neg_l = []
            for i in range(len(p_idx)):
                c = p_cls[i].item()
                pool = clusters[c]
                if len(pool) < 2: n = random.randint(0, n_auth-1)
                else: 
                    n = random.choice(pool)
                    if n == p_idx[i].item(): n = random.choice(pool)
                neg_l.append(n)
            neg = torch.tensor(neg_l, device=DEVICE)
        else:
            neg = torch.randint(0, n_auth, (len(tr_edges[0]),), device=DEVICE)
            
        loss = loss_f(anc, pos, ha[neg])
        loss.backward()
        opt.step()
        
        if ep % 5 == 0:
            val_sc = evaluate(model, val_p, full_g, full_g.node_features, full_g.num_nodes, 10)
            trial.report(val_sc, ep)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()
            
    return evaluate(model, all_papers[n_tr+n_val:], full_g, full_g.node_features, full_g.num_nodes, 10)

def run_hpo(max_p, n_tri, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    papers = open_file()
    if not papers: return None, None
    papers = filter_and_preprocess_data(papers, max_p, cache_dir)
    if not papers: return None, None
    
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5))
    study.optimize(lambda t: objective(t, papers), n_trials=n_tri)
    
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete: 
        print("❌ All HPO trials failed. Check data/graph.")
        return None, None
        
    print(f"Best Score: {study.best_value:.4f}")
    return study, papers

# ---------------------------------------------------------
# 7. FINAL TRAINING & ADVANCED CLUSTERING
# ---------------------------------------------------------
def train_final(params, papers):
    print("Training final model...")
    g = build_graph(papers, 'author_id_list', embedding_key)
    for k in g.node_features: g.node_features[k] = g.node_features[k].to(DEVICE)
    for k in g.edge_index: g.edge_index[k] = g.edge_index[k].to(DEVICE)
    
    model = ManualHeteroGNN(EMBEDDING_DIM, params['h'], params['o'], params['dr']).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
    loss_f = nn.TripletMarginLoss(margin=params['mg'])
    
    edges = g.edge_index[('a','w','p')]
    
    model.train()
    for _ in tqdm(range(params['ep']), desc="Final Train"):
        opt.zero_grad()
        out = model(g.node_features, g.edge_index, g.num_nodes)
        anc = out['paper'][edges[1]]; pos = out['author'][edges[0]]
        neg = out['author'][torch.randint(0, g.num_nodes['author'], (edges.shape[1],), device=DEVICE)]
        loss = loss_f(anc, pos, neg)
        loss.backward()
        opt.step()
        
    model.eval()
    with torch.no_grad(): out = model(g.node_features, g.edge_index, g.num_nodes)
    return out['author'].cpu().numpy(), g.node_maps['author']

# --- ADVANCED EM CLUSTERING ---
STOPWORDS = set(['the', 'and', 'of', 'to', 'in', 'a', 'is', 'that', 'for', 'on', 'with', 'as', 'by', 'this', 'are', 'from', 'at', 'be', 'an', 'which', 'or', 'we', 'using', 'based', 'study', 'results', 'paper', 'proposed', 'method', 'analysis', 'model', 'system', 'data', 'approach', 'performance', 'can', 'has', 'have', 'new', 'two', 'via', 'through', 'between', 'such', 'when', 'where', 'during', 'under', 'over', 'review', 'survey', 'application', 'problem', 'time', 'effective', 'different'])

def get_keywords(texts, n=10):
    words = []
    trans = str.maketrans('', '', string.punctuation)
    for t in texts:
        if not t: continue
        w_list = t.lower().translate(trans).split()
        words.extend([w for w in w_list if w not in STOPWORDS and len(w)>2])
    return [w for w, c in Counter(words).most_common(n)]

def run_advanced_em(embs, auth_map, papers, k, n_top, folder):
    if not SKLEARN_AVAILABLE: return
    print(f"\n📂 Advanced EM (K={k}, Top-{n_top}) -> {folder}/")
    if os.path.exists(folder): shutil.rmtree(folder)
    os.makedirs(folder)
    
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=SEED, n_init=3).fit(embs)
    probs = gmm.predict_proba(embs)
    top_n = np.argsort(probs, axis=1)[:, -n_top:][:, ::-1]
    
    idx2aid = {v: k for k, v in auth_map.items()}
    c_mems = defaultdict(list)
    
    # --- METADATA EXTRACTION FIX ---
    auth_meta = {}
    auth2titles = defaultdict(list)
    
    for p in papers:
        t = p.get('title', '')
        
        # 1. Store Title for Keywords
        for aid in p.get('author_id_list', []):
            auth2titles[aid].append(t)
            
        # 2. Extract Name/Org for Output File
        for author_obj in p.get('authors', []):
            # Try to match author_obj to our ID 'aid'
            # Since our ID is the name, we just grab it
            if 'name' in author_obj:
                aid = str(author_obj['name']).strip()
                if aid not in auth_meta:
                    auth_meta[aid] = {
                        'name': aid, # The ID is the name
                        'org': author_obj.get('org', 'Unknown') # Get org correctly
                    }

    for i in range(len(embs)):
        aid = idx2aid[i]
        for c in top_n[i]: c_mems[c].append(aid)
        
    summary = []
    for c in range(k):
        mems = c_mems[c]
        titles = []
        for m in mems: titles.extend(auth2titles[m])
        kwds = ", ".join(get_keywords(titles))
        summary.append(f"Cluster {c}: {kwds}")
        
        with open(os.path.join(folder, f"cluster_{c}.txt"), "w", encoding="utf-8") as f:
            f.write(f"Cluster {c}\nKeywords: {kwds}\nSize: {len(mems)}\n")
            f.write("-"*60 + "\n")
            f.write(f"{'Author Name (ID)':<30} | {'Affiliation'}\n")
            f.write("-"*60 + "\n")
            
            for m in mems:
                meta = auth_meta.get(m, {'name': m, 'org': 'Unknown'})
                f.write(f"{meta['name']:<30} | {meta['org']}\n")
            
    with open(os.path.join(folder, "KEYWORDS.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
    print("✅ Done.")

# ---------------------------------------------------------
# 8. ASSOCIATION RULES
# ---------------------------------------------------------
def run_rules(papers):
    if not MLXTEND_AVAILABLE: return
    print("\n🪙 Association Rules...")
    txns = [p.get('author_id_list') for p in papers if p.get('author_id_list')]
    if not txns: return
    
    te = TransactionEncoder()
    df = pd.DataFrame(te.fit(txns).transform(txns), columns=te.columns_)
    
    freq = apriori(df, min_support=max(0.005, 5/len(txns)), use_colnames=True)
    if freq.empty: print("No patterns."); return
    
    rules = association_rules(freq, metric="confidence", min_threshold=0.5)
    if rules.empty: print("No rules."); return
    
    simple = rules[(rules['antecedents'].apply(len)==1) & (rules['consequents'].apply(len)==1)]
    print(simple.sort_values('confidence', ascending=False)[['antecedents', 'consequents', 'confidence']].head(10))

# ---------------------------------------------------------
# 9. MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    MAX_P = 2000
    TRIALS = 10
    CACHE = "./embedding_cache"
    
    study, papers = run_hpo(MAX_P, TRIALS, CACHE)
    
    if study:
        emb, amap = train_final(study.best_params, papers)
        
        # 1. Hard Clustering (Primary Field)
        run_advanced_em(emb, amap, papers, k=15, n_top=1, folder="em_results_hard")
        
        # 2. Fuzzy Clustering (Collaboration Candidates)
        run_advanced_em(emb, amap, papers, k=15, n_top=3, folder="em_results_soft")
        
        # 3. Rules
        run_rules(papers)
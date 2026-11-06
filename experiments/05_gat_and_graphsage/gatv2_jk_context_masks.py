import numpy as np
import json
import warnings
import random
import sys
import math
import os
import pickle
import time
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# 1. CONFIGURATION (The Champion Setup)
# ---------------------------------------------------------
SIZES_TO_TEST = [10000] 
K_VALUES_TO_TEST = [5, 10, 20]
MASKS_TO_TEST = [0, 1, 2, 3, 'ALL']  

# --- NEW: Grid Search for Edge Dropout ---
EDGE_DROPOUT_RATES = [0.4]

# We lock Social Weight to 0.2 here to prevent a massive combinatorial explosion 
# of logs, since 0.2 is the scientifically justified optimal ratio.
SOCIAL_WEIGHTS = [0.2] 

# Relaxed filtering to prevent dropping 90% of the dataset
MIN_AUTHORS = 2
K_CORE_MIN = 3       

EMBEDDING_TYPE = 'all-mpnet-base-v2'
embedding_key = 'mpnet_embedding'
HIDDEN_DIM = 512     
TIME_DIM = 32
MAX_EPOCHS = 250
LR = 0.0005          
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ---------------------------------------------------------
# 2. DATA LOADING & PIPELINE
# ---------------------------------------------------------
embed_model = None
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    print("❌ CRITICAL: sentence-transformers not installed.")

def load_embedding_model():
    global embed_model
    if not SENTENCE_TRANSFORMER_AVAILABLE: sys.exit(1)
    if embed_model is None:
        print(f"⏳ Loading AI Brain ({EMBEDDING_TYPE})...")
        embed_model = SentenceTransformer(EMBEDDING_TYPE, device=str(DEVICE))

def preprocess_papers_with_embeddings(papers: List[Dict[str, Any]], model_type: str, cache_path: str) -> List[Dict[str, Any]]:
    if os.path.exists(cache_path):
        print(f"   ⚡ Loading cached embeddings from {cache_path}...")
        try:
            with open(cache_path, 'rb') as f: return pickle.load(f)
        except Exception as e:
            print(f"   ⚠️ Cache load failed ({e}). Regenerating...")

    load_embedding_model()
    print(f"   🧠 Generating embeddings for {len(papers)} papers...")
    
    texts = []
    for p in papers:
        title = p.get('title', '')
        abstract = p.get('abstract', '')
        venue = p.get('venue', '')
        keywords = p.get('keywords', [])
        kw_str = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
        
        text_str = f"Title: {title} [SEP] Abstract: {abstract}"
        if venue: text_str += f" [SEP] Venue: {venue}"
        if kw_str: text_str += f" [SEP] Keywords: {kw_str}"
        texts.append(text_str)

    batch_size = 64
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embs.append(embed_model.encode(batch, convert_to_numpy=True, show_progress_bar=False))
    
    full_emb = np.concatenate(embs)
    for i, p in enumerate(papers): p[embedding_key] = full_emb[i]
        
    print(f"   💾 Saving embeddings to {cache_path}...")
    with open(cache_path, 'wb') as f: pickle.dump(papers, f)
    return papers

def extract_paper_author_ids(paper_data: Dict[str, Any]) -> List[str]:
    if 'author_id_list' in paper_data and paper_data['author_id_list']: return paper_data['author_id_list']
    authors = paper_data.get('authors', [])
    if not isinstance(authors, list): return []
    author_ids = []
    for author in authors:
        if isinstance(author, str): author_ids.append(author); continue
        if not isinstance(author, dict): continue
        for key in ['id', 'authorId', 'author_id', 'name']:
            if key in author and author[key] is not None: 
                author_ids.append(str(author[key])); break
    return list(set(author_ids))

def open_file(filename: str = "data/pid_to_info_all.json") -> Optional[List[Dict[str, Any]]]:
    try:
        with open(filename, 'r', encoding='utf-8') as f: raw_data = json.load(f)
        papers_list = list(raw_data.values()) if isinstance(raw_data, dict) else raw_data
        print(f"   ✅ Successfully loaded {len(papers_list)} raw papers from {filename}")
        for i, p in enumerate(papers_list):
            if 'id' not in p or p['id'] is None: p['id'] = f"paper_{i}_{time.time()}"
        return papers_list
    except FileNotFoundError: 
        print(f"❌ Could not find {filename}")
        sys.exit(1)

def filter_and_preprocess_data(papers_list: List[Dict[str, Any]], max_papers: Optional[int] = None,
                                 model_type: str = EMBEDDING_TYPE, cache_dir: str = ".") -> List[Dict[str, Any]]: 
    if not papers_list: return []
    valid_papers, seen_ids = [], set()
    for paper in papers_list:
        paper_id = paper.get('id')
        if paper_id is None or paper_id in seen_ids: continue
        author_ids = extract_paper_author_ids(paper)
        if (paper.get('title') or paper.get('abstract')) and author_ids:
            paper['author_id_list'] = author_ids
            valid_papers.append(paper)
            seen_ids.add(paper_id)
            
    if max_papers is not None and len(valid_papers) > max_papers:
        valid_papers.sort(key=lambda p: int(p.get('year', 0)) if str(p.get('year', 0)).isdigit() else 0, reverse=True)
        valid_papers = valid_papers[:max_papers]

    cache_filename = f"{model_type}_embeddings_{len(valid_papers)}.pkl"
    cache_path = os.path.join(cache_dir, cache_filename)
    return preprocess_papers_with_embeddings(valid_papers, model_type=model_type, cache_path=cache_path)

def plot_author_stats(papers):
    author_counts = [len(p.get('author_id_list', [])) for p in papers]
    plt.figure(figsize=(10, 5))
    plt.hist(author_counts, bins=range(1, 20), color='skyblue', edgecolor='black')
    plt.title("Distribution of Authors per Paper (Raw Data)")
    plt.xlabel("Number of Authors")
    plt.ylabel("Number of Papers")
    plt.grid(axis='y', alpha=0.3)
    plt.show(block=False)
    plt.pause(3) 

def get_k_core_subset(target_size, k=K_CORE_MIN):
    raw_papers = open_file("data/pid_to_info_all.json")
    processed_papers = filter_and_preprocess_data(raw_papers, max_papers=20000) 
    
    print("\n📊 Plotting author distribution before k-core filtering...")
    plot_author_stats(processed_papers)
    
    current_papers = processed_papers
    
    for _ in range(15):
        auth_counts = Counter()
        for p in current_papers:
            for a in p['author_id_list']: auth_counts[a] += 1
        
        next_papers = []
        for p in current_papers:
            strong = [a for a in p['author_id_list'] if auth_counts[a] >= k]
            if len(strong) >= MIN_AUTHORS:
                p['clean_authors'] = strong
                next_papers.append(p)
                
        if len(next_papers) == len(current_papers): break
        current_papers = next_papers
    
    if len(current_papers) > target_size: current_papers = current_papers[:target_size]
    print(f"   📊 Final dense graph size: {len(current_papers)} papers.")
    return current_papers

# ---------------------------------------------------------
# 3. CORE ARCHITECTURE (MLP Projections + Simple JK)
# ---------------------------------------------------------
class HarmonicTimeEncoder(nn.Module):
    def __init__(self, time_dim=32):
        super().__init__()
        self.time_dim = time_dim
        self.freqs = nn.Parameter(torch.randn(1, time_dim // 2))

    def forward(self, years):
        t = (years - 2010.0) / 10.0 
        angles = t * self.freqs * 2 * math.pi
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

class MultiHeadGATv2BipartiteLayer(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.15):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.W_src = nn.Linear(dim, dim, bias=False)
        self.W_dst = nn.Linear(dim, dim, bias=False)
        
        self.a = nn.Parameter(torch.empty(1, num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.a)
        
        self.attn_dropout = nn.Dropout(dropout)
        
    def forward(self, src_feat, dst_feat, src_idx, dst_idx):
        h_src = self.W_src(src_feat).view(-1, self.num_heads, self.head_dim)
        h_dst = self.W_dst(dst_feat).view(-1, self.num_heads, self.head_dim)
        
        edge_feat = F.leaky_relu(h_src[src_idx] + h_dst[dst_idx], negative_slope=0.2)
        scores = (edge_feat * self.a).sum(dim=-1)
        
        scores = torch.clamp(scores, min=-20, max=20) 
        weights = torch.exp(scores)
        
        sum_weights = torch.zeros(dst_feat.size(0), self.num_heads, device=dst_feat.device)
        sum_weights.index_add_(0, dst_idx, weights)
        
        alpha = weights / (sum_weights[dst_idx] + 1e-10)
        alpha = self.attn_dropout(alpha)
        
        msg = h_src[src_idx] * alpha.unsqueeze(-1)
        
        out = torch.zeros(dst_feat.size(0), self.num_heads, self.head_dim, device=dst_feat.device)
        out.index_add_(0, dst_idx, msg)
        
        return out.view(-1, self.dim)

class Robust2LayerGAT_JK(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads=8):
        super().__init__()
        
        self.proj_p = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.proj_a = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer1_A2P = MultiHeadGATv2BipartiteLayer(hidden_dim, num_heads=num_heads)
        self.layer1_P2A = MultiHeadGATv2BipartiteLayer(hidden_dim, num_heads=num_heads)
        
        self.layer2_A2P = MultiHeadGATv2BipartiteLayer(hidden_dim, num_heads=num_heads)
        self.layer2_P2A = MultiHeadGATv2BipartiteLayer(hidden_dim, num_heads=num_heads)
        
        self.norm_p1 = nn.LayerNorm(hidden_dim)
        self.norm_a1 = nn.LayerNorm(hidden_dim)
        self.norm_p2 = nn.LayerNorm(hidden_dim)
        self.norm_a2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, p_feats, a_feats, edge_index):
        src, dst = edge_index 
        
        h_p_0 = self.proj_p(p_feats)
        h_a_0 = self.proj_a(a_feats)
        
        h_p_1 = self.norm_p1(h_p_0 + self.dropout(self.layer1_A2P(h_a_0, h_p_0, src, dst)))
        h_a_1 = self.norm_a1(h_a_0 + self.dropout(self.layer1_P2A(h_p_0, h_a_0, dst, src)))
        
        h_p_2 = self.norm_p2(h_p_1 + self.dropout(self.layer2_A2P(h_a_1, h_p_1, src, dst)))
        h_a_2 = self.norm_a2(h_a_1 + self.dropout(self.layer2_P2A(h_p_1, h_a_1, dst, src)))
        
        out_p = h_p_0 + h_p_1 + h_p_2
        out_a = h_a_0 + h_a_1 + h_a_2
        
        return F.normalize(out_p, p=2, dim=1), F.normalize(out_a, p=2, dim=1), F.normalize(h_p_0, p=2, dim=1), F.normalize(h_a_0, p=2, dim=1)

# ---------------------------------------------------------
# 4. TRAINING & EVALUATION (Grid Search Added)
# ---------------------------------------------------------
def drop_edges(edge_index, drop_prob=0.5): 
    """Randomly drops edges to simulate cold-start and force feature reliance."""
    if drop_prob == 0.0: return edge_index
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges, device=edge_index.device) > drop_prob
    return edge_index[:, mask]

def train_and_eval():
    print(f"\n🚀 STARTING EXPERIMENT: Multi-Head 8 + MLP Projections + Edge Dropout Grid + 30% Aux Loss")
    
    for size in SIZES_TO_TEST:
        print(f"\n{'='*60}\nTARGET DATASET SIZE: {size}\n{'='*60}")
        papers = get_k_core_subset(size)
        if len(papers) < 50:
            print("❌ Not enough papers. Try a lower K_CORE_MIN."); continue

        random.shuffle(papers)
        
        train_split = int(len(papers) * 0.8)
        train_p = papers[:train_split]
        test_p = papers[train_split:]
        
        auth_set = set()
        for p in train_p: auth_set.update(p['clean_authors'])
        author_map = {n: i for i, n in enumerate(auth_set)}
        idx2name = {i: n for n, i in author_map.items()}
        
        dim = len(papers[0][embedding_key])
        combined_dim = dim + TIME_DIM
        
        p_tensor_text = torch.tensor(np.stack([p[embedding_key] for p in train_p]), device=DEVICE)
        
        p_years = []
        for p in train_p:
            y = p.get('year')
            p_years.append(float(y) if y and str(y).isdigit() else 2020.0)
        p_year_tensor = torch.tensor(p_years, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        
        src_list, dst_list = [], []
        counts = torch.zeros((len(author_map), 1), device=DEVICE)
        for i, p in enumerate(train_p):
            for a in p['clean_authors']:
                idx = author_map[a]
                src_list.append(idx); dst_list.append(i)
                counts[idx] += 1
        
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=DEVICE)
        src_tensor = edge_index[0]  
        dst_tensor = edge_index[1]  
        counts = counts.clamp(min=1)
        
        # --- NEW: Grid Search Loop Wrapper ---
        for drop_rate in EDGE_DROPOUT_RATES:
            print(f"\n{'*'*60}\n🔥 TRAINING WITH EDGE DROPOUT RATE: {drop_rate * 100}%\n{'*'*60}")
            
            # 1. Initialize fresh models for each drop rate so weights don't carry over
            time_encoder = HarmonicTimeEncoder(TIME_DIM).to(DEVICE)
            model = Robust2LayerGAT_JK(combined_dim, HIDDEN_DIM, num_heads=8).to(DEVICE)
            
            logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.05)).to(DEVICE)
            opt = torch.optim.AdamW(list(model.parameters()) + [logit_scale], lr=LR, weight_decay=1e-4)
            opt_time = torch.optim.AdamW(time_encoder.parameters(), lr=LR)
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-5)
            scheduler_time = torch.optim.lr_scheduler.CosineAnnealingLR(opt_time, T_max=MAX_EPOCHS, eta_min=1e-5)
            
            for ep in range(MAX_EPOCHS):
                model.train()
                opt.zero_grad()
                opt_time.zero_grad()
                
                time_embs = time_encoder(p_year_tensor)
                p_tensor = torch.cat([p_tensor_text, time_embs], dim=1)
                
                a_init = torch.zeros((len(author_map), combined_dim), device=DEVICE)
                a_init.index_add_(0, src_tensor, p_tensor[dst_tensor])
                a_init = a_init / counts
                
                # 2. Use the dynamic drop_rate from the loop
                train_edge_index = drop_edges(edge_index, drop_prob=drop_rate)
                
                h_p_final, h_a_final, h_p_0, h_a_0 = model(p_tensor, a_init, train_edge_index)
                temp = torch.clamp(logit_scale.exp(), max=100) 
                
                logits_p2a_main = torch.mm(h_p_final[dst_tensor], h_a_final.t()) * temp
                loss_main_p2a = F.cross_entropy(logits_p2a_main, src_tensor)
                
                logits_a2p_main = torch.mm(h_a_final[src_tensor], h_p_final.t()) * temp
                loss_main_a2p = F.cross_entropy(logits_a2p_main, dst_tensor)
                
                loss_main = (loss_main_p2a + loss_main_a2p) / 2.0
                
                logits_p2a_aux = torch.mm(h_p_0[dst_tensor], h_a_0.t()) * temp
                loss_aux_p2a = F.cross_entropy(logits_p2a_aux, src_tensor)
                
                logits_a2p_aux = torch.mm(h_a_0[src_tensor], h_p_0.t()) * temp
                loss_aux_a2p = F.cross_entropy(logits_a2p_aux, dst_tensor)
                
                loss_aux = (loss_aux_p2a + loss_aux_a2p) / 2.0
                loss = loss_main + (0.3 * loss_aux)
                
                loss.backward()
                opt.step()
                opt_time.step()
                
                scheduler.step()
                scheduler_time.step()
                
                if (ep + 1) % 50 == 0:  # Suppressed logging slightly to avoid terminal spam
                    print(f"       Ep {ep+1:>3}/{MAX_EPOCHS}: Total Loss {loss.item():.4f} (Main: {loss_main.item():.4f}, Aux: {loss_aux.item():.4f})")
                    
            print(f"\n   🎯 Evaluating Test Set for Dropout = {drop_rate} ...")
            model.eval()
            with torch.no_grad():
                time_embs = time_encoder(p_year_tensor)
                p_tensor = torch.cat([p_tensor_text, time_embs], dim=1)
                a_init = torch.zeros((len(author_map), combined_dim), device=DEVICE)
                a_init.index_add_(0, src_tensor, p_tensor[dst_tensor])
                a_init = a_init / counts
                # IMPORTANT: Note that evaluation happens on the FULL edge_index, not the dropped one
                _, final_a_embs, _, _ = model(p_tensor, a_init, edge_index)
                
            print(f"\n{'Drop%':<6} | {'Soc_W':<6} | {'Mask':<4} | {'K':<3} | {'NDCG':<6} | {'Recall':<6}")
            print("-" * 50)
            
            for sw in SOCIAL_WEIGHTS:
                cw = 1.0 - sw
                for mask_n in MASKS_TO_TEST:
                    for k in K_VALUES_TO_TEST:
                        ndcg_scores, rec_scores = [], []
                        for p in test_p:
                            valid = [a for a in p['clean_authors'] if a in author_map]
                            
                            if mask_n == 'ALL' or mask_n == 0:
                                if len(valid) == 0: continue
                                hidden, known = valid, []
                            else:
                                if len(valid) <= mask_n: continue
                                random.shuffle(valid)
                                hidden, known = valid[:mask_n], valid[mask_n:]
                            
                            raw_p_vec = torch.tensor(p[embedding_key], device=DEVICE).unsqueeze(0)
                            y = p.get('year')
                            p_y = float(y) if y and str(y).isdigit() else 2020.0
                            test_year_tensor = torch.tensor([[p_y]], dtype=torch.float32, device=DEVICE)
                            
                            with torch.no_grad():
                                test_time_emb = time_encoder(test_year_tensor)
                                p_vec = torch.cat([raw_p_vec, test_time_emb], dim=1) 
                                
                                h_p_init = model.proj_p(p_vec) 
                                
                                if known:
                                    k_idx = torch.tensor([author_map[a] for a in known], dtype=torch.long, device=DEVICE)
                                    k_feats = model.proj_a(a_init[k_idx])
                                    src_dummy = torch.arange(len(k_idx), device=DEVICE)
                                    dst_dummy = torch.zeros(len(k_idx), dtype=torch.long, device=DEVICE)
                                    
                                    msg1 = model.layer1_A2P(k_feats, h_p_init, src_dummy, dst_dummy)
                                    h_p_1 = model.norm_p1(h_p_init + msg1)
                                    
                                    msg2 = model.layer2_A2P(final_a_embs[k_idx], h_p_1, src_dummy, dst_dummy)
                                    h_p_2 = model.norm_p2(h_p_1 + msg2)
                                    
                                    h_p_final = h_p_init + h_p_1 + h_p_2
                                    h_p_proj = F.normalize(h_p_final, p=2, dim=1)
                                    content_scores = torch.mm(h_p_proj, final_a_embs.t()).squeeze(0)
                                    
                                    social_vec = F.normalize(torch.mean(final_a_embs[k_idx], dim=0, keepdim=True), p=2, dim=1)
                                    social_scores = torch.mm(social_vec, final_a_embs.t()).squeeze(0)
                                    
                                    final_scores = cw * content_scores + sw * social_scores
                                    
                                    for ki in k_idx: final_scores[ki] = -999.0 
                                else:
                                    h_p_proj = F.normalize(h_p_init, p=2, dim=1)
                                    content_scores = torch.mm(h_p_proj, final_a_embs.t()).squeeze(0)
                                    final_scores = content_scores 
                                    
                                topk = torch.topk(final_scores, k).indices.cpu().numpy()
                                preds = [idx2name[i] for i in topk]
                                
                                hits = len(set(preds).intersection(set(hidden)))
                                rec_scores.append(hits / len(hidden))
                                
                                dcg, idcg = 0.0, 0.0
                                for i, name in enumerate(preds):
                                    if name in hidden: dcg += 1.0 / math.log2(i + 2)
                                for i in range(min(len(hidden), k)): idcg += 1.0 / math.log2(i + 2)
                                ndcg_scores.append(dcg/idcg if idcg > 0 else 0)
                                
                        avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0
                        avg_rec = np.mean(rec_scores) if rec_scores else 0
                        print(f"{drop_rate:<6.1f} | {sw:<6.1f} | {mask_n:<4} | {k:<3} | {avg_ndcg:<6.4f} | {avg_rec:<6.4f}")

if __name__ == "__main__":
    train_and_eval()
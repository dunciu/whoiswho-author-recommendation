import numpy as np
import json
import random
import sys
import math
import os
import pickle
import time
from typing import List, Dict, Any, Optional
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# 1. CONFIGURATION & HARDWARE SETUP
# ---------------------------------------------------------
SIZES_TO_TEST = [float('inf')]  # Running on the entire dataset
K_VALUES_TO_TEST = [5, 10, 20]
CONTEXT_LEVELS = [0, 1, 2] 
EDGE_DROPOUT_RATES = [0.0, 0.2] 
SOCIAL_WEIGHTS = [0.2] 

MIN_AUTHORS = 3
K_CORE_MIN = 3       

EMBEDDING_TYPE = 'allenai/specter2_base'
embedding_key = 'mpnet_embedding'
HIDDEN_DIM = 256     # Reduced from 512 for massive memory savings
TIME_DIM = 32
MAX_EPOCHS = 250
LR = 0.0005          
SEED = 42

# --- PyTorch Device Setup ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"🚀 PyTorch Training: Nvidia GPU detected ({torch.cuda.get_device_name(0)})")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ PyTorch Training: No Nvidia GPU found. GNN will train on CPU.")
# ------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ---------------------------------------------------------
# 2. DATA LOADING & EMBEDDING PIPELINE
# ---------------------------------------------------------
embed_model = None
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    print("❌ CRITICAL: sentence-transformers not installed.")
    sys.exit(1)

def load_embedding_model():
    global embed_model
    if embed_model is None:
        print("⏳ Booting up SentenceTransformer for Embeddings...")
        try:
            embed_model = SentenceTransformer(EMBEDDING_TYPE)
            print("🚀 SentenceTransformer active!")
        except Exception as e:
            print(f"❌ Failed to load embedding model. Error: {e}")
            sys.exit(1)

def preprocess_papers_with_embeddings(papers: List[Dict[str, Any]], model_type: str, cache_path: str) -> List[Dict[str, Any]]:
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f: 
                print(f"✅ Loaded cached embeddings from {cache_path}")
                return pickle.load(f)
        except Exception:
            print("⚠️ Cache corrupted, regenerating...")

    load_embedding_model()
    texts = []
    for p in papers:
        title = p.get('title', '')
        abstract = p.get('abstract', '')
        venue = p.get('venue', '')
        keywords = p.get('keywords', [])
        kw_str = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
        texts.append(f"Title: {title} [SEP] Abstract: {abstract} [SEP] Venue: {venue} [SEP] Keywords: {kw_str}")

    print(f"⏳ Generating embeddings for {len(texts)} papers...")
    full_emb = embed_model.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=True)
    
    for i, p in enumerate(papers): 
        p[embedding_key] = full_emb[i]
        
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

    cache_filename = f"{model_type}_embeddings_{len(valid_papers)}.pkl".replace("/", "_")
    cache_path = os.path.join(cache_dir, cache_filename)
    return preprocess_papers_with_embeddings(valid_papers, model_type=model_type, cache_path=cache_path)

def get_k_core_subset(target_size, k=K_CORE_MIN):
    raw_papers = open_file("data/pid_to_info_all.json")
    processed_papers = filter_and_preprocess_data(raw_papers, max_papers=None) 
    
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
    return current_papers

# ---------------------------------------------------------
# 3. UPGRADED CORE ARCHITECTURE (MEMORY SAFE)
# ---------------------------------------------------------
def bpr_loss(pos_scores, neg_scores):
    return -F.logsigmoid(pos_scores - neg_scores).mean()

def contextual_borda_fusion(gnn_s, sim_s, rule_s, ctx):
    if ctx == 0: w_g, w_s, w_r = 0.2, 0.8, 0.0
    elif ctx == 1: w_g, w_s, w_r = 0.3, 0.3, 0.4
    else: w_g, w_s, w_r = 0.3, 0.2, 0.5
        
    N = gnn_s.size(0)
    def get_points(scores):
        ranks = torch.argsort(torch.argsort(scores, descending=True))
        points = (N - ranks.float()) / N
        points[scores == -float('inf')] = -float('inf')
        return points
        
    combined = (w_g * get_points(gnn_s)) + (w_s * get_points(sim_s)) + (w_r * get_points(rule_s))
    combined[gnn_s == -float('inf')] = -float('inf') 
    return combined

class HarmonicTimeEncoder(nn.Module):
    def __init__(self, time_dim=32):
        super().__init__()
        self.time_dim = time_dim
        self.freqs = nn.Parameter(torch.randn(1, time_dim // 2))

    def forward(self, years):
        t = (years - 2010.0) / 10.0 
        angles = t * self.freqs * 2 * math.pi
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


# --- REPLACED GAT WITH MEMORY-EFFICIENT GRAPHSAGE ---
class BipartiteSAGE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, src_feat, dst_feat, src_idx, dst_idx):
        # Gather messages directly (avoids E x Heads tensors)
        msg = src_feat[src_idx]
        
        # Aggregate via Summation
        agg = torch.zeros_like(dst_feat)
        agg.index_add_(0, dst_idx, msg)
        
        # Normalize by Degree (Mean Pooling)
        deg = torch.zeros(dst_feat.size(0), 1, device=dst_feat.device)
        ones = torch.ones(src_idx.size(0), 1, dtype=torch.float32, device=dst_feat.device)
        deg.index_add_(0, dst_idx, ones)
        
        agg = agg / deg.clamp(min=1)
        
        # Update Node State
        out = self.W(torch.cat([dst_feat, agg], dim=-1))
        return self.norm(F.relu(out))

class MemorySafeBipartiteSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.proj_p = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden_dim, hidden_dim))
        self.proj_a = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden_dim, hidden_dim))
        
        # GraphSAGE Layers (Fraction of the memory of GAT)
        self.layer1_A2P = BipartiteSAGE(hidden_dim)
        self.layer1_P2A = BipartiteSAGE(hidden_dim)
        
        self.layer2_A2P = BipartiteSAGE(hidden_dim)
        self.layer2_P2A = BipartiteSAGE(hidden_dim)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, p_feats, a_feats, edge_index_ap):
        src_ap, dst_ap = edge_index_ap 
        
        h_p_0 = self.proj_p(p_feats)
        h_a_0 = self.proj_a(a_feats)
        
        # Pass 1: Authors to Papers, Papers to Authors
        h_p_1 = self.layer1_A2P(h_a_0, h_p_0, src_ap, dst_ap)
        h_a_1 = self.layer1_P2A(h_p_0, h_a_0, dst_ap, src_ap)
        
        # Pass 2: Second hop communication
        h_p_2 = self.layer2_A2P(self.dropout(h_a_1), self.dropout(h_p_1), src_ap, dst_ap)
        h_a_2 = self.layer2_P2A(self.dropout(h_p_1), self.dropout(h_a_1), dst_ap, src_ap)
        
        out_p = h_p_0 + h_p_1 + h_p_2
        out_a = h_a_0 + h_a_1 + h_a_2
        
        return F.normalize(out_p, p=2, dim=1), F.normalize(out_a, p=2, dim=1), F.normalize(h_p_0, p=2, dim=1), F.normalize(h_a_0, p=2, dim=1)

def drop_edges(edge_index, drop_prob=0.4): 
    if drop_prob == 0.0: return edge_index
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges, device=edge_index.device) > drop_prob
    return edge_index[:, mask]

def calculate_ndcg_recall(scores, hidden, K, idx2name):
    if not hidden: return 0.0, 0.0
    topk_idx = torch.topk(scores, K).indices.cpu().numpy()
    preds = [idx2name[i] for i in topk_idx]
    rec = len(set(preds).intersection(set(hidden))) / len(hidden)
    dcg = sum([1.0 / math.log2(i + 2) for i, name in enumerate(preds) if name in hidden])
    idcg = sum([1.0 / math.log2(i + 2) for i in range(min(len(hidden), K))])
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return ndcg, rec

# ---------------------------------------------------------
# 4. TRAINING, EVALUATION & PLOTTING
# ---------------------------------------------------------
def train_and_eval():
    print(f"\n🚀 STARTING FULL EVALUATION (Contrastive GNN + Meta-Ensemble)")
    
    dropout_plot_data = {'ndcg': [], 'recall': []}
    
    for size in SIZES_TO_TEST:
        papers = get_k_core_subset(size)
        if len(papers) < 50: continue

        papers.sort(key=lambda p: float(p.get('year')) if p.get('year') and str(p.get('year')).isdigit() else 2020.0)
        train_split = int(len(papers) * 0.8)
        train_p = papers[:train_split]
        test_p = papers[train_split:]
        
        auth_set = set()
        for p in train_p: auth_set.update(p['clean_authors'])
        author_map = {n: i for i, n in enumerate(auth_set)}
        idx2name = {i: n for n, i in author_map.items()}
        num_authors = len(author_map)
        
        dim = len(papers[0][embedding_key])
        combined_dim = dim + TIME_DIM
        
        p_tensor_text = torch.tensor(np.stack([p[embedding_key] for p in train_p]), device=DEVICE)
        
        p_years = []
        for p in train_p:
            y = p.get('year')
            p_years.append(float(y) if y and str(y).isdigit() else 2020.0)
        p_year_tensor = torch.tensor(p_years, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        years_tensor = torch.tensor(p_years, device=DEVICE)
        
        print("⏳ Building Memory-Safe Synthetic Reference Graph...")
        augmented_p_tensor = p_tensor_text.clone()
        chunk_size = 1000
        for start_idx in range(0, len(train_p), chunk_size):
            end_idx = min(start_idx + chunk_size, len(train_p))
            chunk_embs = F.normalize(p_tensor_text[start_idx:end_idx], p=2, dim=1)
            all_embs = F.normalize(p_tensor_text, p=2, dim=1)
            sim_chunk = torch.mm(chunk_embs, all_embs.t())
            
            chunk_years = years_tensor[start_idx:end_idx].unsqueeze(1)
            all_years = years_tensor.unsqueeze(0)
            time_mask = all_years < chunk_years
            sim_chunk.masked_fill_(~time_mask, -float('inf'))
            
            num_refs = min(30, len(train_p))
            topk_vals, topk_indices = torch.topk(sim_chunk, k=num_refs, dim=1)
            
            for i in range(end_idx - start_idx):
                global_idx = start_idx + i
                valid_mask = topk_vals[i] != -float('inf')
                valid_refs = topk_indices[i][valid_mask]
                if len(valid_refs) > 0:
                    ref_embs = p_tensor_text[valid_refs].mean(dim=0)
                    augmented_p_tensor[global_idx] = (0.8 * p_tensor_text[global_idx]) + (0.2 * ref_embs)

        p_tensor_text = augmented_p_tensor
        print("✅ Synthetic citations mapped and embedded!")
        
        # Force a memory sweep before GNN compilation
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # --- MEMORY EFFICIENT GRAPH BUILDING ---
        src_list, dst_list = [], []
        counts = torch.zeros((num_authors, 1), device=DEVICE)
        global_freq = torch.zeros(num_authors, device=DEVICE)
        
        co_occurrence_dict = Counter()
        
        for i, p in enumerate(train_p):
            a_idx = []
            for a in p['clean_authors']:
                if a in author_map:
                    idx = author_map[a]
                    src_list.append(idx)
                    dst_list.append(i)
                    counts[idx] += 1
                    a_idx.append(idx)
            
            for u in a_idx:
                global_freq[u] += 1
                for v in a_idx:
                    if u != v: 
                        co_occurrence_dict[(u, v)] += 1
        
        edge_index_ap = torch.tensor([src_list, dst_list], dtype=torch.long, device=DEVICE)
        src_tensor = edge_index_ap[0]  
        dst_tensor = edge_index_ap[1]  
        counts = counts.clamp(min=1)
        
        aa_src, aa_dst, aa_weights = [], [], []
        for (u, v), count in co_occurrence_dict.items():
            aa_src.append(u)
            aa_dst.append(v)
            aa_weights.append(count)
            
        aa_src_tensor = torch.tensor(aa_src, dtype=torch.long, device=DEVICE)
        aa_dst_tensor = torch.tensor(aa_dst, dtype=torch.long, device=DEVICE)
        raw_weights = torch.tensor(aa_weights, dtype=torch.float32, device=DEVICE)
        
        aa_weight_tensor = torch.log1p(raw_weights) 
        aa_weight_tensor = (aa_weight_tensor / aa_weight_tensor.max()) * 5.0 
        # ------------------------------------------------

        static_test_p = []
        for p in test_p:
            valid_auths = [a for a in p['clean_authors'] if a in author_map]
            if len(valid_auths) >= MIN_AUTHORS:
                random.shuffle(valid_auths) 
                p['valid_authors'] = valid_auths
                static_test_p.append(p)
                
        print(f"Original Test Papers: {len(test_p)}")
        print(f"Filtered Static Test Papers (>= {MIN_AUTHORS} authors): {len(static_test_p)}\n")
        
        for drop_rate in EDGE_DROPOUT_RATES:
            print(f"\n{'='*130}\n🔥 TRAINING DROPOUT: {drop_rate * 100:.0f}%\n{'='*130}")
            
            time_encoder = HarmonicTimeEncoder(TIME_DIM).to(DEVICE)
            model = MemorySafeBipartiteSAGE(combined_dim, HIDDEN_DIM).to(DEVICE)
            
            logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.05), dtype=torch.float32, device=DEVICE))
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
                
                a_init = torch.zeros((num_authors, combined_dim), device=DEVICE)
                a_init.index_add_(0, src_tensor, p_tensor[dst_tensor])
                a_init = a_init / counts
                
                train_edge_index_ap = drop_edges(edge_index_ap, drop_prob=drop_rate)
                
                h_p_final, h_a_final, h_p_0, h_a_0 = model(p_tensor, a_init, train_edge_index_ap)
                
                temp = torch.clamp(logit_scale.exp(), max=100) 
                
                pos_scores_main = (h_p_final[dst_tensor] * h_a_final[src_tensor]).sum(dim=1) * temp
                pos_scores_aux = (h_p_0[dst_tensor] * h_a_0[src_tensor]).sum(dim=1) * temp
                
                neg_authors = torch.randint(0, num_authors, (len(src_tensor),), device=DEVICE)
                neg_scores_main = (h_p_final[dst_tensor] * h_a_final[neg_authors]).sum(dim=1) * temp
                neg_scores_aux = (h_p_0[dst_tensor] * h_a_0[neg_authors]).sum(dim=1) * temp
                
                loss_main = bpr_loss(pos_scores_main, neg_scores_main)
                loss_aux = bpr_loss(pos_scores_aux, neg_scores_aux)
                loss = loss_main + (0.3 * loss_aux)

                loss.backward()
                opt.step()
                opt_time.step()
                scheduler.step()
                scheduler_time.step()
                
                if ep % 20 == 0:
                    print(f"Epoch {ep:03d} | Loss: {loss.item():.4f}")
                
            model.eval()
            with torch.no_grad():
                time_embs = time_encoder(p_year_tensor)
                p_tensor = torch.cat([p_tensor_text, time_embs], dim=1)
                a_init = torch.zeros((num_authors, combined_dim), device=DEVICE)
                a_init.index_add_(0, src_tensor, p_tensor[dst_tensor])
                a_init = a_init / counts
                
                _, final_a_embs, _, _ = model(p_tensor, a_init, edge_index_ap)
                
            print(f"\n{'Ctx':<3} | {'K':<2} | {'G_NDCG':<6} | {'G_Rec':<6} | {'B_NDCG':<6} | {'B_Rec':<6} | {'META_C':<6} | {'META_R':<6} | {'S_NDCG':<6} | {'S_Rec':<6} | {'R_NDCG':<6} | {'R_Rec':<6}")
            print("-" * 115)
            
            graph_ndcg = 0
            graph_rec = 0
            
            for sw in SOCIAL_WEIGHTS:
                cw = 1.0 - sw
                for ctx in CONTEXT_LEVELS:
                    for K in K_VALUES_TO_TEST:
                        
                        g_ndcg, g_rec = [], [] 
                        b_ndcg, b_rec = [], [] 
                        s_ndcg, s_rec = [], [] 
                        r_ndcg, r_rec = [], [] 
                        c_ndcg, c_rec = [], [] 
                        
                        for p in static_test_p[:1000]: 
                            valid = p['valid_authors'] 
                            known = valid[:ctx]
                            hidden = valid[ctx:]
                            
                            raw_p_vec = torch.tensor(p[embedding_key], device=DEVICE).unsqueeze(0)
                            
                            y = p.get('year')
                            try:
                                test_year = float(y) if y and str(y).isdigit() else 2020.0
                            except:
                                test_year = 2020.0
                                
                            test_year_tensor = torch.tensor([[test_year]], dtype=torch.float32, device=DEVICE)
                            
                            with torch.no_grad():
                                test_time_emb = time_encoder(test_year_tensor)
                                p_vec = torch.cat([raw_p_vec, test_time_emb], dim=1) 
                                h_p_init = model.proj_p(p_vec) 
                                
                                k_idx = torch.tensor([author_map[a] for a in known], dtype=torch.long, device=DEVICE) if known else []
                                
                                # 1. GNN PREDICTIONS
                                if known:
                                    k_feats = model.proj_a(a_init[k_idx])
                                    src_dummy = torch.arange(len(k_idx), device=DEVICE)
                                    dst_dummy = torch.zeros(len(k_idx), dtype=torch.long, device=DEVICE)
                                    
                                    msg1 = model.layer1_A2P(k_feats, h_p_init, src_dummy, dst_dummy)
                                    h_p_1 = msg1
                                    msg2 = model.layer2_A2P(final_a_embs[k_idx], h_p_1, src_dummy, dst_dummy)
                                    h_p_2 = msg2
                                    
                                    h_p_proj = F.normalize(h_p_init + h_p_1 + h_p_2, p=2, dim=1)
                                    content_scores = torch.mm(h_p_proj, final_a_embs.t()).squeeze(0)
                                    
                                    social_vec = F.normalize(torch.mean(final_a_embs[k_idx], dim=0, keepdim=True), p=2, dim=1)
                                    social_scores = torch.mm(social_vec, final_a_embs.t()).squeeze(0)
                                    
                                    gnn_scores = cw * content_scores + sw * social_scores
                                    gnn_scores[k_idx] = -float('inf')
                                else:
                                    h_p_proj = F.normalize(h_p_init, p=2, dim=1)
                                    gnn_scores = torch.mm(h_p_proj, final_a_embs.t()).squeeze(0)
                                
                                # 2. BASELINE 1: RULE-BASED 
                                rule_scores = torch.zeros(num_authors, device=DEVICE)
                                if known:
                                    for ki in k_idx: 
                                        mask = (aa_src_tensor == ki)
                                        neighbors = aa_dst_tensor[mask]
                                        w = aa_weight_tensor[mask]
                                        rule_scores.index_add_(0, neighbors, w)
                                    rule_scores[k_idx] = -float('inf') 
                                else:
                                    rule_scores = global_freq.clone() 

                                # 3. BASELINE 2: MOST SIMILAR PAPERS
                                sim_scores = torch.zeros(num_authors, device=DEVICE)
                                test_emb_norm = F.normalize(raw_p_vec, p=2, dim=1)
                                train_emb_norm = F.normalize(p_tensor_text, p=2, dim=1)
                                
                                paper_sims = torch.mm(test_emb_norm, train_emb_norm.t()).squeeze(0)
                                top_sim_papers = torch.topk(paper_sims, k=20).indices
                                
                                for p_idx in top_sim_papers:
                                    weight = paper_sims[p_idx]
                                    a_indices = src_tensor[dst_tensor == p_idx]
                                    sim_scores[a_indices] += weight
                                    
                                if known: sim_scores[k_idx] = -float('inf')

                                # 4. BASELINE 3: REFERENCE GRAPH
                                ref_scores = torch.zeros(num_authors, device=DEVICE)
                                time_mask = years_tensor < test_year
                                historical_sims = paper_sims.clone()
                                historical_sims.masked_fill_(~time_mask, -float('inf')) 
                                
                                top_cited_papers = torch.topk(historical_sims, k=20).indices
                                for p_idx in top_cited_papers:
                                    if historical_sims[p_idx] != -float('inf'):
                                        weight = historical_sims[p_idx]
                                        a_indices = src_tensor[dst_tensor == p_idx]
                                        ref_scores[a_indices] += weight
                                        
                                if known: ref_scores[k_idx] = -float('inf')

                                combined_scores = contextual_borda_fusion(gnn_scores, sim_scores, rule_scores, ctx)

                                ndcg, rec = calculate_ndcg_recall(gnn_scores, hidden, K, idx2name)
                                g_ndcg.append(ndcg); g_rec.append(rec)
                                
                                ndcg, rec = calculate_ndcg_recall(rule_scores, hidden, K, idx2name)
                                b_ndcg.append(ndcg); b_rec.append(rec)
                                
                                ndcg, rec = calculate_ndcg_recall(combined_scores, hidden, K, idx2name)
                                c_ndcg.append(ndcg); c_rec.append(rec)

                                ndcg, rec = calculate_ndcg_recall(sim_scores, hidden, K, idx2name)
                                s_ndcg.append(ndcg); s_rec.append(rec)
                                
                                ndcg, rec = calculate_ndcg_recall(ref_scores, hidden, K, idx2name)
                                r_ndcg.append(ndcg); r_rec.append(rec)
                                
                        print(f"{str(ctx):<3} | {K:<2} | {np.mean(g_ndcg):<6.4f} | {np.mean(g_rec):<6.4f} | {np.mean(b_ndcg):<6.4f} | {np.mean(b_rec):<6.4f} | {np.mean(c_ndcg):<6.4f} | {np.mean(c_rec):<6.4f} | {np.mean(s_ndcg):<6.4f} | {np.mean(s_rec):<6.4f} | {np.mean(r_ndcg):<6.4f} | {np.mean(r_rec):<6.4f}")
                        
                        if ctx == 0 and K == 20:
                            graph_ndcg = np.mean(g_ndcg)
                            graph_rec = np.mean(g_rec)
            
            dropout_plot_data['ndcg'].append(graph_ndcg)
            dropout_plot_data['recall'].append(graph_rec)

if __name__ == "__main__":
    train_and_eval()
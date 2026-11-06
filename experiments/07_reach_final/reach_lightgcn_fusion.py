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
# 1. OPTIMIZED HYPERPARAMETERS & SETUP
# ---------------------------------------------------------
LR = 0.0005          
DROPOUT_RATE = 0.2   
BATCH_SIZE = 16384   

OPTIMIZED_CONFIGS = {
    0: {"sw": 0.1}, 
    1: {"sw": 0.5}, 
    2: {"sw": 0.7}  
}

K_VALUES = [5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500]
CONTEXT_LEVELS = [0, 1, 2] 
MIN_AUTHORS = 3
K_CORE_MIN = 3       

EMBEDDING_TYPE = 'allenai/specter2_base'
embedding_key = 'mpnet_embedding'
HIDDEN_DIM = 256     
TIME_DIM = 32
MAX_EPOCHS = 200 
SEED = 42

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"🚀 PyTorch: Nvidia GPU detected ({torch.cuda.get_device_name(0)})")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ PyTorch: No Nvidia GPU found. Training on CPU.")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ---------------------------------------------------------
# 2. DATA PIPELINE & K-CORE PRUNING
# ---------------------------------------------------------
embed_model = None

def load_embedding_model():
    global embed_model
    if embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("⏳ Booting up SentenceTransformer...")
            embed_model = SentenceTransformer(EMBEDDING_TYPE)
        except ImportError:
            print("❌ CRITICAL: sentence-transformers not installed.")
            sys.exit(1)

def preprocess_papers_with_embeddings(papers: List[Dict[str, Any]], cache_path: str) -> List[Dict[str, Any]]:
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f: 
            print(f"✅ Loaded cached embeddings from {cache_path}")
            return pickle.load(f)

    load_embedding_model()
    texts = [f"Title: {p.get('title','')} [SEP] Abstract: {p.get('abstract','')}" for p in papers]
    print(f"⏳ Generating embeddings for {len(texts)} papers...")
    full_emb = embed_model.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=True)
    
    for i, p in enumerate(papers): p[embedding_key] = full_emb[i]
    with open(cache_path, 'wb') as f: pickle.dump(papers, f)
    return papers

def extract_paper_author_ids(paper_data: Dict[str, Any]) -> List[str]:
    if 'author_id_list' in paper_data and paper_data['author_id_list']: return paper_data['author_id_list']
    authors = paper_data.get('authors', [])
    if not isinstance(authors, list): return []
    author_ids = []
    for author in authors:
        if isinstance(author, str): 
            author_ids.append(author)
            continue
        if not isinstance(author, dict): continue
        for key in ['id', 'authorId', 'author_id', 'name']:
            if key in author and author[key] is not None: 
                author_ids.append(str(author[key]))
                break
    return list(set(author_ids))

def get_safe_year(p_dict):
    val = p_dict.get('year', 2020)
    if not str(val).strip(): return 2020.0
    try: return float(val)
    except: return 2020.0

def load_base_data():
    try:
        with open("data/pid_to_info_all.json", 'r', encoding='utf-8') as f: raw_data = json.load(f)
    except:
        print("❌ Could not find pid_to_info_all.json")
        sys.exit(1)
        
    papers_list = list(raw_data.values()) if isinstance(raw_data, dict) else raw_data
    valid_papers = []
    for i, p in enumerate(papers_list):
        if 'id' not in p: p['id'] = f"p_{i}"
        a_ids = extract_paper_author_ids(p)
        if a_ids:
            p['author_id_list'] = a_ids
            valid_papers.append(p)
    return valid_papers

def run_k_core_pruning(current_papers, k=K_CORE_MIN):
    print("📉 K-CORE PRUNING TRACKER (AVALANCHE SIMULATION)")
    print(f"Initial Valid Papers (before pruning): {len(current_papers)}")
    
    for iteration in range(15):
        auth_counts = Counter(a for p in current_papers for a in p['author_id_list'])
        next_papers = []
        for p in current_papers:
            strong = [a for a in p['author_id_list'] if auth_counts[a] >= k]
            if len(strong) >= MIN_AUTHORS:
                p['clean_authors'] = strong
                next_papers.append(p)
        
        removed = len(current_papers) - len(next_papers)
        print(f"   Iteration {iteration + 1:>2}: {len(next_papers)} papers remain (Removed: {removed})")
        
        if len(next_papers) == len(current_papers): 
            print(f"✅ Network successfully converged at Iteration {iteration + 1}!")
            return current_papers, len(auth_counts)
        current_papers = next_papers
        
    return current_papers, len(Counter(a for p in current_papers for a in p['author_id_list']))

# ---------------------------------------------------------
# 3. ARCHITECTURE: LIGHTGCN & FUSION HELPERS
# ---------------------------------------------------------
def infonce_loss(pos_scores, neg_scores_matrix):
    all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores_matrix], dim=1)
    labels = torch.zeros(pos_scores.size(0), dtype=torch.long, device=pos_scores.device)
    return F.cross_entropy(all_scores, labels)

def normalize_score(s):
    mask = (s != -float('inf'))
    if not mask.any(): return torch.zeros_like(s)
    s_min, s_max = s[mask].min(), s[mask].max()
    if s_max == s_min: return torch.zeros_like(s)
    norm_s = torch.zeros_like(s)
    norm_s[mask] = (s[mask] - s_min) / (s_max - s_min)
    return norm_s

def get_raw_stats(s):
    mask = (s != -float('inf'))
    if not mask.any(): 
        return 0.0, 0.0
    valid_scores = s[mask]
    return valid_scores.max().item(), valid_scores.std().item() if valid_scores.numel() > 1 else 0.0

def weighted_score_fusion(gnn_s, sim_s, rule_s, weights, ctx=None, exclude=None):
    w_g, w_s, w_r = weights[0], weights[1], weights[2]
    
    if exclude == 'S':
        w_s = torch.zeros_like(w_s)
        total_w = w_g + w_r
        w_g = w_g / (total_w + 1e-8)
        w_r = w_r / (total_w + 1e-8)
    elif exclude == 'B':
        w_r = torch.zeros_like(w_r)
        total_w = w_g + w_s
        w_g = w_g / (total_w + 1e-8)
        w_s = w_s / (total_w + 1e-8)

    if ctx == 0:
        total_w = w_g + w_s
        w_g = w_g / (total_w + 1e-8)
        w_s = w_s / (total_w + 1e-8)
        w_r = torch.zeros_like(w_r)

    norm_g = normalize_score(gnn_s)
    norm_s = normalize_score(sim_s)
    norm_r = normalize_score(rule_s)
    
    combined = (w_g * norm_g) + (w_s * norm_s) + (w_r * norm_r)
    combined[gnn_s == -float('inf')] = -float('inf')
    return combined

class HarmonicTimeEncoder(nn.Module):
    def __init__(self, time_dim=32):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(1, time_dim // 2))
    def forward(self, years):
        t = (years - 2010.0) / 10.0 
        angles = t * self.freqs * 2 * math.pi
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

class BipartiteLightGCNLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, src_f, dst_f, src_idx, dst_idx):
        msg = src_f[src_idx]
        src_deg = torch.zeros(src_f.size(0), 1, device=src_f.device).index_add_(0, src_idx, torch.ones_like(src_idx).float().unsqueeze(1)).clamp(min=1)
        dst_deg = torch.zeros(dst_f.size(0), 1, device=dst_f.device).index_add_(0, dst_idx, torch.ones_like(dst_idx).float().unsqueeze(1)).clamp(min=1)
        norm = 1.0 / torch.sqrt(src_deg[src_idx] * dst_deg[dst_idx])
        msg = msg * norm
        agg = torch.zeros_like(dst_f).index_add_(0, dst_idx, msg)
        return agg

class MemorySafeLightGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.proj_p = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.proj_a = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.layer = BipartiteLightGCNLayer()
        self.dropout = nn.Dropout(0.2)

    def forward(self, p_feats, a_feats, edge_index_ap):
        src_ap, dst_ap = edge_index_ap 
        h_p_0, h_a_0 = self.proj_p(p_feats), self.proj_a(a_feats)
        h_p_1 = self.layer(h_a_0, h_p_0, src_ap, dst_ap)
        h_a_1 = self.layer(h_p_0, h_a_0, dst_ap, src_ap)
        h_p_2 = self.layer(self.dropout(h_a_1), self.dropout(h_p_1), src_ap, dst_ap)
        h_a_2 = self.layer(self.dropout(h_p_1), self.dropout(h_a_1), dst_ap, src_ap)
        out_p = (h_p_0 + h_p_1 + h_p_2) / 3.0
        out_a = (h_a_0 + h_a_1 + h_a_2) / 3.0
        return F.normalize(out_p, p=2, dim=1), F.normalize(out_a, p=2, dim=1), F.normalize(h_p_0, p=2, dim=1), F.normalize(h_a_0, p=2, dim=1)

def drop_edges(edge_index, drop_prob): 
    if drop_prob == 0.0: return edge_index
    mask = torch.rand(edge_index.size(1), device=edge_index.device) > drop_prob
    return edge_index[:, mask]

def calculate_ndcg_recall(scores, hidden, K, idx2name):
    if not hidden: return 0.0, 0.0
    topk_idx = torch.topk(scores, K).indices.cpu().numpy()
    preds = [idx2name[i] for i in topk_idx]
    rec = len(set(preds).intersection(set(hidden))) / len(hidden)
    dcg = sum([1.0 / math.log2(i + 2) for i, name in enumerate(preds) if name in hidden])
    idcg = sum([1.0 / math.log2(i + 2) for i in range(min(len(hidden), K))])
    return (dcg / idcg if idcg > 0 else 0.0), rec

# ---------------------------------------------------------
# 4. ENCAPSULATED PIPELINE EXPERIMENT
# ---------------------------------------------------------
def run_experiment(raw_papers, split_name, cache_name):
    print("\n" + "="*80)
    print(f"🌟 INITIALIZING EXPERIMENT: {split_name}")
    print("="*80)
    
    pruned_papers, total_authors = run_k_core_pruning(raw_papers)
    
    print("-" * 50)
    print(f"🎯 Final Robust Dataset Size: {len(pruned_papers)} papers")
    print(f"👥 Total Unique Authors: {total_authors}")
    print("-" * 50)
    
    if len(pruned_papers) < 100:
        print(f"⚠️ Warning: Dataset '{split_name}' is too small after pruning. Skipping.")
        return

    papers = preprocess_papers_with_embeddings(pruned_papers, cache_name)
    papers.sort(key=get_safe_year)
    train_split = int(len(papers) * 0.8)
    train_p, test_p = papers[:train_split], papers[train_split:]
    
    auth_set = set(a for p in train_p for a in p['clean_authors'])
    author_map = {n: i for i, n in enumerate(auth_set)}
    idx2name = {i: n for n, i in author_map.items()}
    num_authors = len(author_map)
    combined_dim = len(papers[0][embedding_key]) + TIME_DIM
    
    p_tensor_text = torch.tensor(np.stack([p[embedding_key] for p in train_p]), device=DEVICE)
    years_tensor = torch.tensor([get_safe_year(p) for p in train_p], device=DEVICE)
    p_year_tensor = years_tensor.unsqueeze(1)
    
    print(f"📊 Training Set: {len(train_p)} Papers | Testing Set: {len(test_p)} Papers")

    print("⏳ Building Fast Memory-Safe Graph & Rule Dictionaries...")
    src_list, dst_list = [], []
    counts = torch.zeros((num_authors, 1), device=DEVICE)
    global_freq = torch.zeros(num_authors, device=DEVICE)
    co_occurrence_dict = Counter()
    co_occurrence_years = {} 
    
    for i, p in enumerate(train_p):
        a_idx = [author_map[a] for a in p['clean_authors'] if a in author_map]
        p_year = get_safe_year(p)
        for idx in a_idx:
            src_list.append(idx); dst_list.append(i); counts[idx] += 1
        for u in a_idx:
            global_freq[u] += 1
            for v in a_idx:
                if u != v: 
                    co_occurrence_dict[(u, v)] += 1
                    if (u, v) not in co_occurrence_years or p_year > co_occurrence_years[(u, v)]:
                        co_occurrence_years[(u, v)] = p_year
    
    edge_index_ap = torch.tensor([src_list, dst_list], device=DEVICE)
    src_tensor, dst_tensor = edge_index_ap[0], edge_index_ap[1]
    counts = counts.clamp(min=1)
    
    aa_src, aa_dst, aa_weights, aa_years = [], [], [], []
    for (u, v), c in co_occurrence_dict.items():
        aa_src.append(u); aa_dst.append(v); aa_weights.append(c)
        aa_years.append(co_occurrence_years[(u, v)])

    aa_src_tensor = torch.tensor(aa_src, device=DEVICE)
    aa_dst_tensor = torch.tensor(aa_dst, device=DEVICE)
    raw_weight = torch.log1p(torch.tensor(aa_weights, dtype=torch.float32, device=DEVICE))
    aa_weight_tensor = (raw_weight / raw_weight.max()) * 5.0 
    aa_year_tensor = torch.tensor(aa_years, dtype=torch.float32, device=DEVICE) 

    static_test_p = [p for p in test_p if len([a for a in p['clean_authors'] if a in author_map]) >= MIN_AUTHORS]
    for p in static_test_p: random.shuffle(p['clean_authors'])
    
    print("⏳ Mining 2-Hop Hard Negatives (Friends of Friends)...")
    adj_list = {i: set() for i in range(num_authors)}
    for (u, v) in co_occurrence_dict.keys():
        adj_list[u].add(v); adj_list[v].add(u)
        
    MAX_HARD_POOL = 64
    hard_neg_pool = torch.randint(0, num_authors, (num_authors, MAX_HARD_POOL), device=DEVICE)
    for u in range(num_authors):
        friends = adj_list[u]
        fof = set()
        for f in friends: fof.update(adj_list[f])
        fof -= friends
        fof.discard(u)
        fof_list = list(fof)
        if len(fof_list) > 0:
            chosen = np.random.choice(fof_list, MAX_HARD_POOL, replace=True)
            hard_neg_pool[u] = torch.tensor(chosen, device=DEVICE)

    print(f"\n🔥 TRAINING LIGHTGCN & HARMONIC ENCODER [{split_name}]")
    time_encoder = HarmonicTimeEncoder(TIME_DIM).to(DEVICE)
    model = MemorySafeLightGCN(combined_dim, HIDDEN_DIM).to(DEVICE)
    logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.05), device=DEVICE))
    
    opt = torch.optim.AdamW(list(model.parameters()) + list(time_encoder.parameters()) + [logit_scale], lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-5)
    
    popular_authors_dist = global_freq / global_freq.sum()
    num_total_edges = len(src_tensor)
    
    for ep in range(MAX_EPOCHS):
        model.train(); opt.zero_grad()
        p_tensor = torch.cat([p_tensor_text, time_encoder(p_year_tensor)], dim=1)
        a_init = torch.zeros((num_authors, combined_dim), device=DEVICE).index_add_(0, src_tensor, p_tensor[dst_tensor]) / counts
        
        train_edges = drop_edges(edge_index_ap, DROPOUT_RATE)
        h_p_final, h_a_final, h_p_0, h_a_0 = model(p_tensor, a_init, train_edges)
        
        batch_size = min(BATCH_SIZE, num_total_edges)
        batch_idx = torch.randperm(num_total_edges, device=DEVICE)[:batch_size]
        batch_src, batch_dst = src_tensor[batch_idx], dst_tensor[batch_idx]

        temp = torch.clamp(logit_scale.exp(), max=100) 
        pos_scores = (h_p_final[batch_dst] * h_a_final[batch_src]).sum(dim=1) * temp
        
        NUM_HARD, NUM_RAND, NUM_POP = 20, 22, 22
        rand_neg = torch.randint(0, num_authors, (batch_size, NUM_RAND), device=DEVICE)
        pop_neg_1d = torch.multinomial(popular_authors_dist, batch_size * NUM_POP, replacement=True)
        pop_neg = pop_neg_1d.view(batch_size, NUM_POP)
        hard_idx = torch.randint(0, MAX_HARD_POOL, (batch_size, NUM_HARD), device=DEVICE)
        hard_neg = torch.gather(hard_neg_pool[batch_src], 1, hard_idx)
        
        neg_authors_matrix = torch.cat([rand_neg, pop_neg, hard_neg], dim=1) 
        neg_scores = (h_p_final[batch_dst].unsqueeze(1) * h_a_final[neg_authors_matrix]).sum(dim=2) * temp 
        
        pos_scores_0 = (h_p_0[batch_dst] * h_a_0[batch_src]).sum(dim=1) * temp
        neg_scores_0 = (h_p_0[batch_dst].unsqueeze(1) * h_a_0[neg_authors_matrix]).sum(dim=2) * temp
        
        loss = infonce_loss(pos_scores, neg_scores) + (0.3 * infonce_loss(pos_scores_0, neg_scores_0))
               
        loss.backward(); opt.step(); scheduler.step()
        if ep % 50 == 0: print(f"   Epoch {ep:03d} | Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        p_tensor = torch.cat([p_tensor_text, time_encoder(p_year_tensor)], dim=1)
        a_init = torch.zeros((num_authors, combined_dim), device=DEVICE).index_add_(0, src_tensor, p_tensor[dst_tensor]) / counts
        _, final_a_embs, _, _ = model(p_tensor, a_init, edge_index_ap)

    # 5. Fast Static Meta-Fusion Training (UPGRADED: Score-Aware Architecture + Ablation Variants)
    print(f"\n🧠 CACHING STATIC EXPERT SCORES FOR NEURAL FUSION [{split_name}]")
    
    def get_fusion_block():
        return nn.Sequential(
            nn.Linear(combined_dim + 3 + 6, 64), 
            nn.ReLU(), 
            nn.Linear(64, 3), 
            nn.Softmax(dim=-1)
        ).to(DEVICE)
        
    fusion_net = get_fusion_block()
    fusion_net_no_s = get_fusion_block()
    fusion_net_no_b = get_fusion_block()
    
    fusion_opt = torch.optim.Adam(fusion_net.parameters(), lr=0.001)
    fusion_opt_no_s = torch.optim.Adam(fusion_net_no_s.parameters(), lr=0.001)
    fusion_opt_no_b = torch.optim.Adam(fusion_net_no_b.parameters(), lr=0.001)
    
    val_candidates = [p for p in train_p if len([a for a in p['clean_authors'] if a in author_map]) >= MIN_AUTHORS]
    val_papers = random.sample(val_candidates, min(400, len(val_candidates)))
    
    val_dataset = []
    for p in val_papers:
        valid = [a for a in p['clean_authors'] if a in author_map]
        for ctx in [0, 1, 2]:
            known, hidden = valid[:ctx], valid[ctx:]
            if not hidden: continue
            
            raw_p_vec = torch.tensor(p[embedding_key], device=DEVICE).unsqueeze(0)
            test_year_val = get_safe_year(p)
            
            with torch.no_grad():
                p_vec = torch.cat([raw_p_vec, time_encoder(torch.tensor([[test_year_val]], device=DEVICE))], dim=1) 
                h_p_init = model.proj_p(p_vec) 
                content_scores = torch.mm(F.normalize(h_p_init, p=2, dim=1), final_a_embs.t()).squeeze()
                
                if ctx == 0:
                    paper_sims = torch.mm(F.normalize(raw_p_vec, p=2, dim=1), F.normalize(p_tensor_text, p=2, dim=1).t()).squeeze()
                    topk_p_idxs = torch.topk(paper_sims, k=5).indices
                    bootstrapped_authors = []
                    for p_idx in topk_p_idxs: bootstrapped_authors.extend(src_tensor[dst_tensor == p_idx].tolist())
                    bootstrapped_authors = list(set(bootstrapped_authors))
                    
                    if len(bootstrapped_authors) > 0:
                        b_k_idx = torch.tensor(bootstrapped_authors, device=DEVICE)
                        social_scores = torch.mm(F.normalize(torch.mean(final_a_embs[b_k_idx], dim=0, keepdim=True), p=2, dim=1), final_a_embs.t()).squeeze()
                        gnn_scores = 0.4 * content_scores + 0.6 * social_scores
                    else:
                        gnn_scores = content_scores
                    rule_scores = global_freq.clone()
                    k_idx = []
                else:
                    k_idx = torch.tensor([author_map[a] for a in known], device=DEVICE)
                    cw, sw = 1.0 - OPTIMIZED_CONFIGS[ctx]["sw"], OPTIMIZED_CONFIGS[ctx]["sw"]
                    social_scores = torch.mm(F.normalize(torch.mean(final_a_embs[k_idx], dim=0, keepdim=True), p=2, dim=1), final_a_embs.t()).squeeze()
                    gnn_scores = cw * content_scores + sw * social_scores
                    gnn_scores[k_idx] = -float('inf')
                    
                    rule_scores = torch.zeros(num_authors, device=DEVICE)
                    for ki in k_idx: 
                        mask = (aa_src_tensor == ki)
                        time_diff = torch.clamp(test_year_val - aa_year_tensor[mask], min=0)
                        rule_scores.index_add_(0, aa_dst_tensor[mask], aa_weight_tensor[mask] * torch.exp(-0.138 * time_diff))
                    rule_scores[k_idx] = -float('inf') 
                
                sim_scores = torch.zeros(num_authors, device=DEVICE)
                paper_sims = torch.mm(F.normalize(raw_p_vec, p=2, dim=1), F.normalize(p_tensor_text, p=2, dim=1).t()).squeeze()
                for p_idx in torch.topk(paper_sims, k=20).indices: sim_scores[src_tensor[dst_tensor == p_idx]] += paper_sims[p_idx]
                if len(k_idx) > 0: sim_scores[k_idx] = -float('inf')

                g_max, g_std = get_raw_stats(gnn_scores)
                s_max, s_std = get_raw_stats(sim_scores)
                r_max, r_std = get_raw_stats(rule_scores)
                score_stats = torch.tensor([g_max, g_std, s_max, s_std, r_max, r_std], device=DEVICE)

                ctx_onehot = torch.zeros(3, device=DEVICE); ctx_onehot[ctx] = 5.0 
                input_feats = torch.cat([p_vec.squeeze(0), ctx_onehot, score_stats]).detach()
                hidden_tensor = torch.tensor([author_map[a] for a in hidden], device=DEVICE)
                
                val_dataset.append({
                    'input_feats': input_feats, 'gnn_scores': gnn_scores,
                    'sim_scores': sim_scores, 'rule_scores': rule_scores,
                    'hidden_tensor': hidden_tensor, 'ctx': ctx
                })

    for ep in range(30):
        fusion_net.train(); fusion_net_no_s.train(); fusion_net_no_b.train()
        ep_loss, ep_loss_ns, ep_loss_nb = 0.0, 0.0, 0.0
        random.shuffle(val_dataset)
        for item in val_dataset:
            # Full Model REACH
            dynamic_weights = fusion_net(item['input_feats'])
            meta_scores = weighted_score_fusion(item['gnn_scores'], item['sim_scores'], item['rule_scores'], dynamic_weights, ctx=item['ctx'])
            loss = -F.log_softmax(meta_scores * 10.0, dim=0)[item['hidden_tensor']].mean()
            fusion_opt.zero_grad(); loss.backward(); fusion_opt.step()
            ep_loss += loss.item()
            
            # Ablation: No S
            dw_ns = fusion_net_no_s(item['input_feats'])
            meta_scores_ns = weighted_score_fusion(item['gnn_scores'], item['sim_scores'], item['rule_scores'], dw_ns, ctx=item['ctx'], exclude='S')
            loss_ns = -F.log_softmax(meta_scores_ns * 10.0, dim=0)[item['hidden_tensor']].mean()
            fusion_opt_no_s.zero_grad(); loss_ns.backward(); fusion_opt_no_s.step()
            ep_loss_ns += loss_ns.item()

            # Ablation: No B
            dw_nb = fusion_net_no_b(item['input_feats'])
            meta_scores_nb = weighted_score_fusion(item['gnn_scores'], item['sim_scores'], item['rule_scores'], dw_nb, ctx=item['ctx'], exclude='B')
            loss_nb = -F.log_softmax(meta_scores_nb * 10.0, dim=0)[item['hidden_tensor']].mean()
            fusion_opt_no_b.zero_grad(); loss_nb.backward(); fusion_opt_no_b.step()
            ep_loss_nb += loss_nb.item()
            
        if ep % 10 == 0: 
            print(f"   Meta-Epoch {ep:02d} | Fusion Loss: {ep_loss/len(val_dataset):.4f} | NoS: {ep_loss_ns/len(val_dataset):.4f} | NoB: {ep_loss_nb/len(val_dataset):.4f}")

    # 6. Fast Single-Pass Evaluation with Micro-Timing Breakdown
    print(f"\n🚀 SINGLE-PASS EVALUATION FOR: {split_name}")
    print("Ctx | K  | G_NDCG | G_Rec  | B_NDCG | B_Rec  | W_NDCG | W_Rec  | META_C | META_R | M_noS_C| M_noS_R| M_noB_C| M_noB_R| S_NDCG | S_Rec  ")
    print("-" * 155)

    fusion_net.eval(); fusion_net_no_s.eval(); fusion_net_no_b.eval()
    eval_metrics = {c: {k: {
        'gn':[], 'gr':[], 'bn':[], 'br':[], 'wn':[], 'wr':[], 'mn':[], 'mr':[], 'sn':[], 'sr':[],
        'mns_n':[], 'mns_r':[], 'mnb_n':[], 'mnb_r':[]
    } for k in K_VALUES} for c in CONTEXT_LEVELS}
    
    # Micro-timing clocks for structural evaluation
    time_stats = {"GNN": 0.0, "RB": 0.0, "SEM": 0.0, "REACH": 0.0}

    for idx, p in enumerate(static_test_p):
        valid = [a for a in p['clean_authors'] if a in author_map]
        raw_p_vec = torch.tensor(p[embedding_key], device=DEVICE).unsqueeze(0)
        test_year_val = get_safe_year(p)
        
        # --- [SEM Engine] Setup Stage ---
        t_start = time.time()
        with torch.no_grad():
            p_vec = torch.cat([raw_p_vec, time_encoder(torch.tensor([[test_year_val]], device=DEVICE))], dim=1) 
            h_p_init = model.proj_p(p_vec) 
            content_scores = torch.mm(F.normalize(h_p_init, p=2, dim=1), final_a_embs.t()).squeeze()
            paper_sims = torch.mm(F.normalize(raw_p_vec, p=2, dim=1), F.normalize(p_tensor_text, p=2, dim=1).t()).squeeze()
        time_stats["SEM"] += (time.time() - t_start)
            
        for ctx in CONTEXT_LEVELS:
            sw, cw = OPTIMIZED_CONFIGS[ctx]["sw"], 1.0 - OPTIMIZED_CONFIGS[ctx]["sw"]
            known, hidden = valid[:ctx], valid[ctx:]
            if not hidden: continue

            # --- [GNN Engine] Score Evaluation ---
            t_start = time.time()
            with torch.no_grad():
                if ctx == 0:
                    topk_p_idxs = torch.topk(paper_sims, k=5).indices
                    bootstrapped_authors = []
                    for p_idx in topk_p_idxs: bootstrapped_authors.extend(src_tensor[dst_tensor == p_idx].tolist())
                    bootstrapped_authors = list(set(bootstrapped_authors))
                    
                    if len(bootstrapped_authors) > 0:
                        b_k_idx = torch.tensor(bootstrapped_authors, device=DEVICE)
                        social_scores = torch.mm(F.normalize(torch.mean(final_a_embs[b_k_idx], dim=0, keepdim=True), p=2, dim=1), final_a_embs.t()).squeeze()
                        gnn_scores = 0.4 * content_scores + 0.6 * social_scores
                    else:
                        gnn_scores = content_scores
                    k_idx = []
                else:
                    k_idx = torch.tensor([author_map[a] for a in known], device=DEVICE)
                    social_scores = torch.mm(F.normalize(torch.mean(final_a_embs[k_idx], dim=0, keepdim=True), p=2, dim=1), final_a_embs.t()).squeeze()
                    gnn_scores = cw * content_scores + sw * social_scores
                    gnn_scores[k_idx] = -float('inf')
            time_stats["GNN"] += (time.time() - t_start)

            # --- [RB Engine] Score Evaluation ---
            t_start = time.time()
            with torch.no_grad():
                if ctx == 0:
                    rule_scores = global_freq.clone()
                else:
                    rule_scores = torch.zeros(num_authors, device=DEVICE)
                    for ki in k_idx: 
                        mask = (aa_src_tensor == ki)
                        rule_scores.index_add_(0, aa_dst_tensor[mask], aa_weight_tensor[mask] * torch.exp(-0.138 * torch.clamp(test_year_val - aa_year_tensor[mask], min=0)))
                    rule_scores[k_idx] = -float('inf') 
            time_stats["RB"] += (time.time() - t_start)

            # --- [SEM Engine] Score Aggregation ---
            t_start = time.time()
            with torch.no_grad():
                sim_scores = torch.zeros(num_authors, device=DEVICE)
                for p_idx in torch.topk(paper_sims, k=20).indices: sim_scores[src_tensor[dst_tensor == p_idx]] += paper_sims[p_idx]
                if len(k_idx) > 0: sim_scores[k_idx] = -float('inf')
            time_stats["SEM"] += (time.time() - t_start)

            # --- [REACH Meta Framework] Dynamic Gating ---
            t_start = time.time()
            with torch.no_grad():
                w_gnn_n = normalize_score(gnn_scores)
                w_sim_n = normalize_score(sim_scores)
                w_rule_n = normalize_score(rule_scores)
                whoiswho_scores = 0.4 * w_gnn_n + 0.4 * w_sim_n + 0.2 * w_rule_n
                if len(k_idx) > 0: whoiswho_scores[k_idx] = -float('inf')

                g_max, g_std = get_raw_stats(gnn_scores)
                s_max, s_std = get_raw_stats(sim_scores)
                r_max, r_std = get_raw_stats(rule_scores)
                score_stats = torch.tensor([g_max, g_std, s_max, s_std, r_max, r_std], device=DEVICE)

                ctx_onehot = torch.zeros(3, device=DEVICE); ctx_onehot[ctx] = 5.0 
                feat_tensor = torch.cat([p_vec.squeeze(0), ctx_onehot, score_stats])
                
                # Full Prediction
                dynamic_weights = fusion_net(feat_tensor)
                meta_scores = weighted_score_fusion(gnn_scores, sim_scores, rule_scores, dynamic_weights, ctx=ctx)

                # Ablation Predictions
                dw_ns = fusion_net_no_s(feat_tensor)
                meta_scores_no_s = weighted_score_fusion(gnn_scores, sim_scores, rule_scores, dw_ns, ctx=ctx, exclude='S')

                dw_nb = fusion_net_no_b(feat_tensor)
                meta_scores_no_b = weighted_score_fusion(gnn_scores, sim_scores, rule_scores, dw_nb, ctx=ctx, exclude='B')
            time_stats["REACH"] += (time.time() - t_start)

            # --- Metrics Append ---
            for K in K_VALUES:
                m = eval_metrics[ctx][K]
                n, r = calculate_ndcg_recall(gnn_scores, hidden, K, idx2name); m['gn'].append(n); m['gr'].append(r)
                n, r = calculate_ndcg_recall(rule_scores, hidden, K, idx2name); m['bn'].append(n); m['br'].append(r)
                n, r = calculate_ndcg_recall(whoiswho_scores, hidden, K, idx2name); m['wn'].append(n); m['wr'].append(r)
                n, r = calculate_ndcg_recall(meta_scores, hidden, K, idx2name); m['mn'].append(n); m['mr'].append(r)
                n, r = calculate_ndcg_recall(meta_scores_no_s, hidden, K, idx2name); m['mns_n'].append(n); m['mns_r'].append(r)
                n, r = calculate_ndcg_recall(meta_scores_no_b, hidden, K, idx2name); m['mnb_n'].append(n); m['mnb_r'].append(r)
                n, r = calculate_ndcg_recall(sim_scores, hidden, K, idx2name); m['sn'].append(n); m['sr'].append(r)

    for ctx in CONTEXT_LEVELS:
        for K in K_VALUES:
            m = eval_metrics[ctx][K]
            print(f"{str(ctx):<3} | {K:<2} | {np.mean(m['gn']):<6.4f} | {np.mean(m['gr']):<6.4f} | "
                  f"{np.mean(m['bn']):<6.4f} | {np.mean(m['br']):<6.4f} | {np.mean(m['wn']):<6.4f} | {np.mean(m['wr']):<6.4f} | "
                  f"{np.mean(m['mn']):<6.4f} | {np.mean(m['mr']):<6.4f} | "
                  f"{np.mean(m['mns_n']):<6.4f} | {np.mean(m['mns_r']):<6.4f} | "
                  f"{np.mean(m['mnb_n']):<6.4f} | {np.mean(m['mnb_r']):<6.4f} | "
                  f"{np.mean(m['sn']):<6.4f} | {np.mean(m['sr']):<6.4f} ")
                  
    print("\n⏱️ INFERENCE RUNNING TIME PROFILE BREAKDOWN BY COMPONENT")
    print("-" * 65)
    for component, duration in time_stats.items():
        print(f" 🔸 {component:<10} Cumulative Inference Phase Duration: {duration:>8.4f} seconds")
    print("-" * 65)

# ---------------------------------------------------------
# 7. MAIN EXECUTION TRIGGER
# ---------------------------------------------------------
if __name__ == "__main__":
    print("⏳ Loading raw data from disk...")
    all_raw_papers = load_base_data()
    
    run_experiment(all_raw_papers, split_name="FULL DATASET", cache_name="embeddings_split_full.pkl")
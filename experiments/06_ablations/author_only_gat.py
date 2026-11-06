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
# CONFIGURATION
# ---------------------------------------------------------
SIZES_TO_TEST = [20000] # Set to your safe limit. Make sure you have a matching .pkl file!
K_VALUES_TO_TEST = [5, 10, 20]
CONTEXT_LEVELS = [0, 1, 2] 
EDGE_DROPOUT_RATES = [0.7]
SOCIAL_WEIGHTS = [0.75] 

MIN_AUTHORS = 3
K_CORE_MIN = 3       

EMBEDDING_TYPE = 'allenai/specter2_base'
embedding_key = 'mpnet_embedding'
HIDDEN_DIM = 512     
TIME_DIM = 32
MAX_EPOCHS = 50
LR = 0.0005          
SEED = 42

if hasattr(torch, "xpu") and torch.xpu.is_available(): DEVICE = torch.device("xpu")
elif torch.cuda.is_available(): DEVICE = torch.device("cuda")
else: DEVICE = torch.device("cpu")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ---------------------------------------------------------
# DATA LOADING PIPELINE
# ---------------------------------------------------------
embed_model = None

def load_embedding_model():
    global embed_model
    if embed_model is None:
        from sentence_transformers import SentenceTransformer
        print("⏳ Booting up Embeddings...")
        embed_model = SentenceTransformer(EMBEDDING_TYPE)

def preprocess_papers_with_embeddings(papers: List[Dict[str, Any]], model_type: str, cache_path: str) -> List[Dict[str, Any]]:
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f: return pickle.load(f)
    load_embedding_model()
    texts = [f"Title: {p.get('title', '')} [SEP] Abstract: {p.get('abstract', '')} [SEP] Venue: {p.get('venue', '')}" for p in papers]
    print(f"⏳ Generating embeddings for {len(texts)} papers...")
    full_emb = embed_model.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=True)
    for i, p in enumerate(papers): p[embedding_key] = full_emb[i]
    with open(cache_path, 'wb') as f: pickle.dump(papers, f)
    return papers

def extract_paper_author_ids(paper_data: Dict[str, Any]) -> List[str]:
    if 'author_id_list' in paper_data and paper_data['author_id_list']: return paper_data['author_id_list']
    authors = paper_data.get('authors', [])
    author_ids = []
    for author in authors:
        if isinstance(author, str): author_ids.append(author); continue
        if not isinstance(author, dict): continue
        for key in ['id', 'authorId', 'author_id', 'name']:
            if key in author and author[key] is not None: 
                author_ids.append(str(author[key])); break
    return list(set(author_ids))

def open_file(filename: str = "data/pid_to_info_all.json"):
    with open(filename, 'r', encoding='utf-8') as f: raw_data = json.load(f)
    papers_list = list(raw_data.values()) if isinstance(raw_data, dict) else raw_data
    for i, p in enumerate(papers_list):
        if 'id' not in p or p['id'] is None: p['id'] = f"paper_{i}_{time.time()}"
    return papers_list

def filter_and_preprocess_data(papers_list, max_papers=None, model_type=EMBEDDING_TYPE, cache_dir="."): 
    # EARLY CACHE CHECK: Bypass all JSON processing if we already have the exact target file.
    if max_papers is not None:
        filename = f"{model_type}_embeddings_{max_papers}.pkl".replace("/", "_")
        expected_cache_path = os.path.join(cache_dir, filename)
        if os.path.exists(expected_cache_path):
            print(f"⚡ Fast-loading {max_papers} papers directly from cache. Bypassing JSON processing.")
            with open(expected_cache_path, 'rb') as f: return pickle.load(f)

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

    filename = f"{model_type}_embeddings_{len(valid_papers)}.pkl".replace("/", "_")
    cache_path = os.path.join(cache_dir, filename)
    return preprocess_papers_with_embeddings(valid_papers, model_type=model_type, cache_path=cache_path)

def get_k_core_subset(target_size, k=K_CORE_MIN):
    # EARLY JSON BYPASS: Don't load the massive raw JSON if the cache exists!
    filename = f"{EMBEDDING_TYPE}_embeddings_{target_size}.pkl".replace("/", "_")
    expected_cache = os.path.join(".", filename)
    
    if os.path.exists(expected_cache):
        print(f"🚀 Bypassing raw JSON load! Jumping straight to cached data...")
        processed_papers = filter_and_preprocess_data([], max_papers=target_size) 
    else:
        print(f"⚠️ Cache missing for {target_size}. Loading raw JSON... this will take a while.")
        raw_papers = open_file("data/pid_to_info_all.json")
        processed_papers = filter_and_preprocess_data(raw_papers, max_papers=target_size) 
    
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
    return current_papers[:target_size]

# ---------------------------------------------------------
# AUTHOR-ONLY MODEL ARCHITECTURE
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

class MultiHeadGATv2HomogeneousLayer(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.15):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.W = nn.Linear(dim, dim, bias=False)
        self.a = nn.Parameter(torch.empty(1, num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.a)
        self.attn_dropout = nn.Dropout(dropout)
        
    def forward(self, feat, src_idx, dst_idx):
        h = self.W(feat).view(-1, self.num_heads, self.head_dim)
        edge_feat = F.leaky_relu(h[src_idx] + h[dst_idx], negative_slope=0.2)
        scores = (edge_feat * self.a).sum(dim=-1)
        scores = torch.clamp(scores, min=-20, max=20) 
        weights = torch.exp(scores)
        sum_weights = torch.zeros(feat.size(0), self.num_heads, device=feat.device)
        sum_weights.index_add_(0, dst_idx, weights)
        alpha = weights / (sum_weights[dst_idx] + 1e-10)
        alpha = self.attn_dropout(alpha)
        msg = h[src_idx] * alpha.unsqueeze(-1)
        out = torch.zeros(feat.size(0), self.num_heads, self.head_dim, device=feat.device)
        out.index_add_(0, dst_idx, msg)
        return out.view(-1, self.dim)

class AuthorOnlyGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads=8):
        super().__init__()
        self.proj_p = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden_dim, hidden_dim))
        self.proj_a = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden_dim, hidden_dim))
        self.layer1_A2A = MultiHeadGATv2HomogeneousLayer(hidden_dim, num_heads=num_heads)
        self.layer2_A2A = MultiHeadGATv2HomogeneousLayer(hidden_dim, num_heads=num_heads)
        self.norm_a1 = nn.LayerNorm(hidden_dim)
        self.norm_a2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, p_feats, a_feats, edge_index_aa):
        h_p_0 = self.proj_p(p_feats)
        h_a_0 = self.proj_a(a_feats)
        src, dst = edge_index_aa 
        h_a_1 = self.norm_a1(h_a_0 + self.dropout(self.layer1_A2A(h_a_0, src, dst)))
        h_a_2 = self.norm_a2(h_a_1 + self.dropout(self.layer2_A2A(h_a_1, src, dst)))
        out_a = h_a_0 + h_a_1 + h_a_2
        return F.normalize(h_p_0, p=2, dim=1), F.normalize(out_a, p=2, dim=1), F.normalize(h_p_0, p=2, dim=1), F.normalize(h_a_0, p=2, dim=1)

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
# TRAINING AND EVALUATION
# ---------------------------------------------------------
def train_and_eval():
    print(f"\n🚀 STARTING AUTHOR-ONLY EVALUATION")
    for size in SIZES_TO_TEST:
        print(f"\n🔍 Fetching subset for size: {size}...")
        papers = get_k_core_subset(size)
        
        print(f"🛑 Papers surviving K-Core filter: {len(papers)}")
        if len(papers) < 50: 
            print("⚠️ Not enough papers left! The K-core filter wiped out the dataset. Skipping.")
            continue

        def get_safe_year(p):
            y = p.get('year')
            return int(y) if y is not None and str(y).isdigit() else 9999 

        papers.sort(key=get_safe_year)
        train_split = int(len(papers) * 0.8)
        train_p = papers[:train_split]
        test_p = papers[train_split:]
        
        auth_set = set()
        for p in train_p: auth_set.update(p['clean_authors'])
        author_map = {n: i for i, n in enumerate(auth_set)}
        idx2name = {i: n for n, i in author_map.items()}
        num_authors = len(author_map)
        
        combined_dim = len(papers[0][embedding_key]) + TIME_DIM
        p_tensor_text = torch.tensor(np.stack([p[embedding_key] for p in train_p]), device=DEVICE)
        years_tensor = torch.tensor([float(p.get('year')) if p.get('year') and str(p.get('year')).isdigit() else 2020.0 for p in train_p], device=DEVICE)
        p_year_tensor = years_tensor.unsqueeze(1).float()

        src_list, dst_list = [], []
        counts = torch.zeros((num_authors, 1), device=DEVICE)
        co_occurrence = torch.zeros((num_authors, num_authors), device=DEVICE)
        global_freq = torch.zeros(num_authors, device=DEVICE)
        
        for i, p in enumerate(train_p):
            a_idx = []
            for a in p['clean_authors']:
                if a in author_map:
                    idx = author_map[a]
                    src_list.append(idx); dst_list.append(i)
                    counts[idx] += 1
                    a_idx.append(idx)
            for u in a_idx:
                global_freq[u] += 1
                for v in a_idx:
                    if u != v: co_occurrence[u, v] += 1
        
        src_tensor = torch.tensor(src_list, dtype=torch.long, device=DEVICE)
        dst_tensor = torch.tensor(dst_list, dtype=torch.long, device=DEVICE)
        counts = counts.clamp(min=1)

        aa_src, aa_dst = co_occurrence.nonzero(as_tuple=True)
        edge_index_aa = torch.stack([aa_src, aa_dst], dim=0)
        edge_index_aa = edge_index_aa[:, edge_index_aa[0] != edge_index_aa[1]]

        static_test_p = []
        for p in test_p:
            valid_auths = [a for a in p['clean_authors'] if a in author_map]
            if len(valid_auths) >= MIN_AUTHORS:
                random.shuffle(valid_auths) 
                p['valid_authors'] = valid_auths
                static_test_p.append(p)
                
        for drop_rate in EDGE_DROPOUT_RATES:
            time_encoder = HarmonicTimeEncoder(TIME_DIM).to(DEVICE)
            model_ao = AuthorOnlyGAT(combined_dim, HIDDEN_DIM, num_heads=8).to(DEVICE)
            logit_scale_ao = nn.Parameter(torch.ones([]) * np.log(1 / 0.05)).to(DEVICE)
            
            opt_ao = torch.optim.AdamW(list(model_ao.parameters()) + [logit_scale_ao, *time_encoder.parameters()], lr=LR, weight_decay=1e-4)
            sched_ao = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ao, T_max=MAX_EPOCHS, eta_min=1e-5)
            
            print(f"\n⚡ Starting Training Loop for {MAX_EPOCHS} epochs...")
            for ep in range(MAX_EPOCHS):
                model_ao.train()
                opt_ao.zero_grad()
                
                time_embs = time_encoder(p_year_tensor)
                p_tensor = torch.cat([p_tensor_text, time_embs], dim=1)
                
                a_init = torch.zeros((num_authors, combined_dim), device=DEVICE)
                a_init.index_add_(0, src_tensor, p_tensor[dst_tensor])
                a_init = a_init / counts
                
                train_edge_index_aa = drop_edges(edge_index_aa, drop_prob=drop_rate)
                h_p_final, h_a_final, h_p_0, h_a_0 = model_ao(p_tensor, a_init, train_edge_index_aa)
                
                neg_authors = torch.randint(0, num_authors, (len(src_tensor),), device=DEVICE)
                temp = torch.clamp(logit_scale_ao.exp(), max=100) 
                
                pos_main = (h_p_final[dst_tensor] * h_a_final[src_tensor]).sum(dim=1) * temp
                pos_aux = (h_p_0[dst_tensor] * h_a_0[src_tensor]).sum(dim=1) * temp
                neg_main = (h_p_final[dst_tensor] * h_a_final[neg_authors]).sum(dim=1) * temp
                neg_aux = (h_p_0[dst_tensor] * h_a_0[neg_authors]).sum(dim=1) * temp
                
                loss = -F.logsigmoid(pos_main - neg_main).mean() + 0.3 * -F.logsigmoid(pos_aux - neg_aux).mean()
                loss.backward()
                opt_ao.step()
                sched_ao.step()
                
                # PROGRESS TRACKER
                if (ep + 1) % 10 == 0 or ep == 0:
                    print(f"🔄 Epoch {ep + 1:03d}/{MAX_EPOCHS} | Loss: {loss.item():.4f}")
                
            model_ao.eval()
            with torch.no_grad():
                time_embs = time_encoder(p_year_tensor)
                p_tensor = torch.cat([p_tensor_text, time_embs], dim=1)
                a_init = torch.zeros((num_authors, combined_dim), device=DEVICE)
                a_init.index_add_(0, src_tensor, p_tensor[dst_tensor])
                a_init = a_init / counts
                _, final_a_embs, _, _ = model_ao(p_tensor, a_init, edge_index_aa)
                
            print(f"\n{'Ctx':<3} | {'K':<2} | {'AO_NDC':<6} | {'AO_Rec':<6} | {'B_NDCG':<6} | {'B_Rec':<6} | {'S_NDCG':<6} | {'S_Rec':<6}")
            print("-" * 135)
            print("⏳ Pre-calculating global test variables...")
            train_emb_norm = F.normalize(p_tensor_text, p=2, dim=1)

            for sw in SOCIAL_WEIGHTS:
                cw = 1.0 - sw
                for ctx in CONTEXT_LEVELS:
                    for K in K_VALUES_TO_TEST:
                        ao_ndcg_list, ao_rec_list = [], [] 
                        b_ndcg, b_rec = [], [] 
                        s_ndcg, s_rec = [], [] 
                        
                        for p in static_test_p[:500]: # Capped for debug speed
                            valid = p['valid_authors'] 
                            known = valid[:ctx]
                            hidden = valid[ctx:]
                            raw_p_vec = torch.tensor(p[embedding_key], device=DEVICE).unsqueeze(0)
                            test_year = float(p.get('year')) if p.get('year') and str(p.get('year')).isdigit() else 2020.0
                            
                            with torch.no_grad():
                                test_time_emb = time_encoder(torch.tensor([[test_year]], dtype=torch.float32, device=DEVICE))
                                p_vec = torch.cat([raw_p_vec, test_time_emb], dim=1) 
                                k_idx = torch.tensor([author_map[a] for a in known], dtype=torch.long, device=DEVICE) if known else []
                                
                                h_p_proj = F.normalize(model_ao.proj_p(p_vec), p=2, dim=1)
                                if known:
                                    social_vec = F.normalize(torch.mean(final_a_embs[k_idx], dim=0, keepdim=True), p=2, dim=1)
                                    ao_scores = cw * torch.mm(h_p_proj, final_a_embs.t()).squeeze(0) + sw * torch.mm(social_vec, final_a_embs.t()).squeeze(0)
                                    ao_scores[k_idx] = -float('inf')
                                else:
                                    ao_scores = torch.mm(h_p_proj, final_a_embs.t()).squeeze(0)

                                rule_scores = torch.zeros(num_authors, device=DEVICE)
                                if known:
                                    for ki in k_idx: rule_scores += co_occurrence[ki]
                                    rule_scores[k_idx] = -float('inf') 
                                else: rule_scores = global_freq.clone() 

                                sim_scores = torch.zeros(num_authors, device=DEVICE)
                                paper_sims = torch.mm(F.normalize(raw_p_vec, p=2, dim=1), train_emb_norm.t()).squeeze(0)
                                for p_idx in torch.topk(paper_sims, k=20).indices:
                                    sim_scores[src_tensor[dst_tensor == p_idx]] += paper_sims[p_idx]
                                if known: sim_scores[k_idx] = -float('inf')

                                ndcg, rec = calculate_ndcg_recall(ao_scores, hidden, K, idx2name)
                                ao_ndcg_list.append(ndcg); ao_rec_list.append(rec)
                                ndcg, rec = calculate_ndcg_recall(rule_scores, hidden, K, idx2name)
                                b_ndcg.append(ndcg); b_rec.append(rec)
                                ndcg, rec = calculate_ndcg_recall(sim_scores, hidden, K, idx2name)
                                s_ndcg.append(ndcg); s_rec.append(rec)
                                
                        print(f"{str(ctx):<3} | {K:<2} | {np.mean(ao_ndcg_list):<6.4f} | {np.mean(ao_rec_list):<6.4f} | {np.mean(b_ndcg):<6.4f} | {np.mean(b_rec):<6.4f} | {np.mean(s_ndcg):<6.4f} | {np.mean(s_rec):<6.4f}")

if __name__ == "__main__":
    train_and_eval()
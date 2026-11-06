"""
REACH v2: Learning-based Paper Authorship Completion on Academic Graphs.

Given a query paper (title, abstract, venue, year) and C in {0, 1, 2} known
authors, rank all training-graph authors by how likely they are to be among
the remaining authors.

Changes from v1 (reach_lightgcn_fusion.py, which produced the draft's Table III):

  1. Leak-free fusion training. The graph is built on the oldest 70% of papers,
     the fusion layer is trained on the next 10% (papers whose edges are NOT in
     the graph), and everything is tested on the newest 20%. v1 trained the
     fusion layer on training papers that were already in the graph, so every
     expert looked unrealistically reliable.
  2. Venue and institution feature augmentation (VIF): SPECTER2 embeddings of
     venue names (paper side) and of each author's historical affiliations
     (author side), reduced with PCA to --aux-dim. Disable with --no-vif.
  3. Temporally decayed adjacency with a learned rate (TEW):
     M[a, p] = exp(-lambda * (T_ref - year_p)), lambda = softplus(param),
     initialised at 0.138. Disable with --no-tew (binary adjacency).
  4. --linear-time ablation: replaces the harmonic time encoding with a
     single linear year feature.
  5. Proper k-core pruning: counts are recomputed over the surviving
     (pruned) author lists each iteration, until convergence.
  6. Stronger baselines: the fixed 0.4/0.4/0.2 WhoIsWho-style ensemble, plus
     a static ensemble whose weights are grid-searched per context level on
     the validation split.
  7. Fusion ablations: REACH without the GNN, SEM or RB expert.
  8. Per-query metrics saved to disk, with paired-bootstrap 95% CIs and
     p-values for REACH vs every baseline, and dataset statistics (Table I).

Run from the repository root:

    python experiments/07_reach_final/reach.py --seed 0
    python experiments/07_reach_final/reach.py --synthetic 3000 --epochs 20   # quick CPU smoke test
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

K_EVAL = [5, 10, 20, 30, 40, 50, 100]
CONTEXT_LEVELS = [0, 1, 2]
# Weight of the social (known-author) term inside the GNN expert, per context level.
# C=0 uses a "bootstrapped" social anchor built from the 5 most similar training papers.
SOCIAL_WEIGHT = {0: 0.6, 1: 0.5, 2: 0.7}
RB_DECAY = 0.138          # fixed decay for the rule-based expert (half-life of ~5 years)
MIN_AUTHORS = 3           # min in-graph authors for a val/test paper (so C=2 still hides >= 1)
FUSION_CAND_K = 500       # candidate pool per expert when training the fusion layer
METHODS = ["GNN", "RB", "SEM", "WHO_FIXED", "STATIC_TUNED", "REACH",
           "REACH_noGNN", "REACH_noSEM", "REACH_noRB"]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/pid_to_info_all.json")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out-dir", default="results/runs")
    ap.add_argument("--tag", default=None, help="run name; default is built from the flags")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-papers", type=int, default=0, help="random subset of raw papers before pruning (0 = all)")
    ap.add_argument("--synthetic", type=int, default=0, help="generate N synthetic papers instead of loading data")
    ap.add_argument("--embed-model", default="allenai/specter2_base")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--edge-dropout", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--time-dim", type=int, default=32)
    ap.add_argument("--aux-dim", type=int, default=128, help="PCA dim for venue / institution embeddings")
    ap.add_argument("--k-core", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--fusion-papers", type=int, default=1500)
    ap.add_argument("--fusion-epochs", type=int, default=30)
    ap.add_argument("--max-test-papers", type=int, default=0, help="cap on test papers (0 = all)")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--no-vif", action="store_true", help="ablation: no venue/institution features")
    ap.add_argument("--no-tew", action="store_true", help="ablation: no temporal edge weighting (lambda = 0)")
    ap.add_argument("--linear-time", action="store_true", help="ablation: linear year instead of harmonic encoding")
    args = ap.parse_args()
    if args.tag is None:
        parts = ["full"] if not (args.no_vif or args.no_tew or args.linear_time) else []
        if args.no_vif: parts.append("noVIF")
        if args.no_tew: parts.append("noTEW")
        if args.linear_time: parts.append("linearTE")
        if args.synthetic: parts.append(f"synthetic{args.synthetic}")
        args.tag = "_".join(parts)
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def safe_year(val, default=2020.0):
    try:
        s = str(val).strip()
        return float(s) if s else default
    except (TypeError, ValueError):
        return default


def inv_softplus(x):
    return math.log(math.expm1(x))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def parse_authors(authors):
    """Return [(author_id, org)] with the same id precedence as v1 (id > authorId > author_id > name)."""
    out, seen = [], set()
    if not isinstance(authors, list):
        return out
    for a in authors:
        if isinstance(a, str):
            aid, org = a.strip(), ""
        elif isinstance(a, dict):
            aid = None
            for key in ("id", "authorId", "author_id", "name"):
                v = a.get(key)
                if v is not None and str(v).strip():
                    aid = str(v).strip()
                    break
            if aid is None:
                continue
            org = a.get("org") if isinstance(a.get("org"), str) else ""
            org = org.strip()
        else:
            continue
        if aid and aid not in seen:
            seen.add(aid)
            out.append((aid, org))
    return out


def load_papers(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        sys.exit(f"Could not find {path}. See data/README.md, or run with --synthetic 3000 for a smoke test.")
    items = raw.values() if isinstance(raw, dict) else raw
    papers = []
    for i, p in enumerate(items):
        if not isinstance(p, dict):
            continue
        auths = parse_authors(p.get("authors", []))
        if not auths:
            continue
        venue = p.get("venue") if isinstance(p.get("venue"), str) else ""
        papers.append({
            "id": str(p.get("id", f"p_{i}")),
            "title": p.get("title") or "",
            "abstract": p.get("abstract") or "",
            "venue": venue.strip(),
            "year": safe_year(p.get("year")),
            "authors": auths,
        })
    return papers


def generate_synthetic(n, seed, dim=64):
    """Community-structured fake papers so the full pipeline can be smoke-tested on CPU."""
    rng = np.random.default_rng(seed)
    n_comm, per_comm = 25, 40
    centroids = rng.normal(size=(n_comm, dim))
    papers = []
    for i in range(n):
        c = int(rng.integers(n_comm))
        k = int(rng.integers(3, 7))
        members = rng.choice(per_comm, size=k, replace=False)
        auths = [(f"a{c}_{m}", f"org_{(c * 7 + m) % 30}") for m in members]
        if rng.random() < 0.2:  # occasional cross-community author
            c2 = int(rng.integers(n_comm))
            auths.append((f"a{c2}_{int(rng.integers(per_comm))}", "org_x"))
        papers.append({
            "id": f"s{i}", "title": "", "abstract": "", "venue": f"venue_{c % 8}",
            "year": float(rng.integers(2010, 2021)),
            "authors": list(dict(auths).items()),
            "text_emb": (centroids[c] + 0.8 * rng.normal(size=dim)).astype(np.float32),
        })
    return papers


def kcore_prune(papers, k):
    """Keep authors with >= k papers and papers with >= k such authors, iterated on the pruned lists."""
    cur = [(p, [a for a, _ in p["authors"]]) for p in papers]
    for it in range(100):
        cnt = Counter(a for _, al in cur for a in al)
        nxt = []
        for p, al in cur:
            strong = [a for a in al if cnt[a] >= k]
            if len(strong) >= k:
                nxt.append((p, strong))
        e_before, e_after = sum(len(al) for _, al in cur), sum(len(s) for _, s in nxt)
        print(f"   k-core iteration {it + 1:>2}: {len(nxt)} papers, {e_after} authorship edges")
        cur = nxt
        if e_after == e_before:
            break
    out = []
    for p, strong in cur:
        sset = set(strong)
        p = dict(p)
        p["clean"] = strong
        p["org_of"] = {a: o for a, o in p["authors"] if a in sset}
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# Embeddings (cached)
# ---------------------------------------------------------------------------
class TextEncoder:
    def __init__(self, model_name, cache_dir, synthetic_dim=0):
        self.model_name, self.cache_dir, self.synthetic_dim = model_name, cache_dir, synthetic_dim
        self.model = None
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, kind, keys):
        h = hashlib.md5(("\n".join(keys) + self.model_name).encode("utf-8")).hexdigest()[:12]
        safe = self.model_name.replace("/", "_")
        return os.path.join(self.cache_dir, f"{kind}_{safe}_{len(keys)}_{h}.npy")

    def encode(self, kind, keys, texts):
        if self.synthetic_dim:
            out = np.zeros((len(texts), self.synthetic_dim), dtype=np.float32)
            for i, t in enumerate(texts):
                seed = int(hashlib.md5(t.encode("utf-8")).hexdigest()[:8], 16)
                out[i] = np.random.default_rng(seed).normal(size=self.synthetic_dim)
            return out
        path = self._cache_path(kind, keys)
        if os.path.exists(path):
            print(f"   loaded cached {kind} embeddings: {path}")
            return np.load(path)
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                sys.exit("sentence-transformers is not installed: pip install -r requirements.txt")
            self.model = SentenceTransformer(self.model_name)
        print(f"   encoding {len(texts)} {kind} strings with {self.model_name} ...")
        emb = self.model.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
        np.save(path, emb)
        return emb


def pca_reduce(X, d, seed, fit_rows=50000):
    """PCA to d dims (fit on a sample), L2-normalised, zero-padded if fewer than d components exist."""
    out = np.zeros((X.shape[0], d), dtype=np.float32)
    if X.shape[0] == 0:
        return out
    rng = np.random.default_rng(seed)
    fit = X if X.shape[0] <= fit_rows else X[rng.choice(X.shape[0], fit_rows, replace=False)]
    mu = fit.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(fit - mu, full_matrices=False)
    k = min(d, vt.shape[0])
    Z = (X - mu) @ vt[:k].T
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
    out[:, :k] = Z
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class HarmonicTimeEncoder(nn.Module):
    def __init__(self, time_dim=32):
        super().__init__()
        self.out_dim = time_dim
        self.freqs = nn.Parameter(torch.randn(1, time_dim // 2))

    def forward(self, years):  # years: (N, 1)
        t = (years - 2010.0) / 10.0
        ang = t * self.freqs * 2 * math.pi
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class LinearTimeEncoder(nn.Module):
    out_dim = 1

    def forward(self, years):
        return (years - 2010.0) / 10.0


class DecayedLightGCN(nn.Module):
    """2-layer bipartite LightGCN on D_A^-1/2 M(T) D_P^-1/2, with M(T)[a,p] = exp(-lambda (T - y_p))."""

    def __init__(self, in_p, in_a, hidden, dropout=0.2, learn_decay=True, init_lambda=RB_DECAY):
        super().__init__()
        self.proj_p = nn.Sequential(nn.Linear(in_p, hidden), nn.LayerNorm(hidden))
        self.proj_a = nn.Sequential(nn.Linear(in_a, hidden), nn.LayerNorm(hidden))
        self.dropout = nn.Dropout(dropout)
        self.learn_decay = learn_decay
        if learn_decay:
            self.raw_lambda = nn.Parameter(torch.tensor(inv_softplus(init_lambda)))

    def decay_rate(self):
        return F.softplus(self.raw_lambda) if self.learn_decay else torch.tensor(0.0)

    def forward(self, xp, xa, a_idx, p_idx, age):
        n_p, n_a = xp.size(0), xa.size(0)
        w = torch.exp(-self.decay_rate() * age) if self.learn_decay else torch.ones_like(age)
        deg_a = torch.zeros(n_a, device=xp.device).index_add_(0, a_idx, w).clamp(min=1e-6)
        deg_p = torch.zeros(n_p, device=xp.device).index_add_(0, p_idx, w).clamp(min=1e-6)
        norm = (w / torch.sqrt(deg_a[a_idx] * deg_p[p_idx])).unsqueeze(1)

        hp0, ha0 = self.proj_p(xp), self.proj_a(xa)
        d = hp0.size(1)

        def a2p(ha):
            return torch.zeros(n_p, d, device=xp.device).index_add_(0, p_idx, ha[a_idx] * norm)

        def p2a(hp):
            return torch.zeros(n_a, d, device=xp.device).index_add_(0, a_idx, hp[p_idx] * norm)

        hp1, ha1 = a2p(ha0), p2a(hp0)
        hp2, ha2 = a2p(self.dropout(ha1)), p2a(self.dropout(hp1))
        hp = (hp0 + hp1 + hp2) / 3.0
        ha = (ha0 + ha1 + ha2) / 3.0
        return F.normalize(hp, dim=1), F.normalize(ha, dim=1), F.normalize(hp0, dim=1), F.normalize(ha0, dim=1)


def infonce(pos, neg):
    logits = torch.cat([pos.unsqueeze(1), neg], dim=1)
    return F.cross_entropy(logits, torch.zeros(pos.size(0), dtype=torch.long, device=pos.device))


def minmax(s):
    """Min-max normalise finite entries to [0, 1]; -inf entries become 0."""
    m = torch.isfinite(s)
    out = torch.zeros_like(s)
    if not m.any():
        return out
    lo, hi = s[m].min(), s[m].max()
    if hi > lo:
        out[m] = (s[m] - lo) / (hi - lo)
    return out


def score_stats(s):
    v = s[torch.isfinite(s)]
    if v.numel() == 0:
        return [0.0, 0.0]
    return [v.max().item(), v.std().item() if v.numel() > 1 else 0.0]


def fuse(norm_stack, w, ctx, exclude=()):
    """norm_stack: (3, N) min-max normalised [GNN, SEM, RB]; w: (3,) weights."""
    keep = torch.tensor(["G" not in exclude, "S" not in exclude, ("B" not in exclude) and ctx != 0],
                        dtype=w.dtype, device=w.device)
    w = w * keep
    w = w / (w.sum() + 1e-8)
    return (w.unsqueeze(1) * norm_stack).sum(0)


def ndcg_recall_prefix(top_idx, hidden_set, ks):
    hits = [1 if i in hidden_set else 0 for i in top_idx]
    res = {}
    for K in ks:
        h = hits[:K]
        dcg = sum(x / math.log2(r + 2) for r, x in enumerate(h))
        idcg = sum(1.0 / math.log2(r + 2) for r in range(min(len(hidden_set), K)))
        res[K] = (dcg / idcg if idcg > 0 else 0.0, sum(h) / len(hidden_set))
    return res


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device()
    run_dir = os.path.join(args.out_dir, f"{args.tag}_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Device: {device} | run: {run_dir}")

    # ---------------- data ----------------
    if args.synthetic:
        raw = generate_synthetic(args.synthetic, args.seed)
    else:
        print(f"Loading {args.data} ...")
        raw = load_papers(args.data)
    if args.max_papers and len(raw) > args.max_papers:
        raw = random.Random(args.seed).sample(raw, args.max_papers)
    print(f"Valid raw papers: {len(raw)}")

    stats = {"raw_papers": len(raw),
             "raw_authors": len({a for p in raw for a, _ in p["authors"]}),
             "raw_edges": sum(len(p["authors"]) for p in raw)}

    papers = kcore_prune(raw, args.k_core)
    del raw
    papers.sort(key=lambda p: (p["year"], p["id"]))
    n = len(papers)
    n_test, n_val = int(n * args.test_frac), int(n * args.val_frac)
    train_p = papers[: n - n_test - n_val]
    val_p = papers[n - n_test - n_val: n - n_test]
    test_p = papers[n - n_test:]

    authors = sorted({a for p in train_p for a in p["clean"]})
    amap = {a: i for i, a in enumerate(authors)}
    n_a = len(authors)

    rng = random.Random(args.seed)

    def eligible(ps):
        out = []
        for p in ps:
            valid = [a for a in p["clean"] if a in amap]
            if len(valid) >= MIN_AUTHORS:
                rng.shuffle(valid)
                out.append((p, valid))
        return out

    val_q, test_q = eligible(val_p), eligible(test_p)
    if args.max_test_papers and len(test_q) > args.max_test_papers:
        test_q = rng.sample(test_q, args.max_test_papers)

    pruned_authors = {a for p in papers for a in p["clean"]}
    pruned_edges = sum(len(p["clean"]) for p in papers)
    stats.update({
        "pruned_papers": n, "pruned_authors": len(pruned_authors), "pruned_edges": pruned_edges,
        "avg_authors_per_paper": round(pruned_edges / max(n, 1), 3),
        "avg_papers_per_author": round(pruned_edges / max(len(pruned_authors), 1), 3),
        "unique_venues": len({p["venue"] for p in papers if p["venue"]}),
        "unique_institutions": len({o for p in papers for o in p["org_of"].values() if o}),
        "year_range": [papers[0]["year"], papers[-1]["year"]] if papers else None,
        "train_papers": len(train_p), "val_papers": len(val_p), "test_papers": len(test_p),
        "candidate_authors": n_a, "val_queries_eligible": len(val_q), "test_queries_eligible": len(test_q),
        "train_year_max": train_p[-1]["year"], "test_year_min": test_p[0]["year"] if test_p else None,
    })
    print(json.dumps(stats, indent=2))
    with open(os.path.join(run_dir, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # ---------------- embeddings ----------------
    enc = TextEncoder(args.embed_model, args.cache_dir, synthetic_dim=64 if args.synthetic else 0)
    if args.synthetic:
        text = np.stack([p["text_emb"] for p in papers])
    else:
        text = enc.encode("text", [p["id"] for p in papers],
                          [f"Title: {p['title']} [SEP] Abstract: {p['abstract']}" for p in papers])
    pid_row = {p["id"]: i for i, p in enumerate(papers)}

    use_vif = not args.no_vif
    if use_vif:
        venues = sorted({p["venue"] for p in papers if p["venue"]})
        v_emb = pca_reduce(enc.encode("venue", venues, venues), args.aux_dim, args.seed)
        v_row = {v: i for i, v in enumerate(venues)}
        venue_feat = np.zeros((n, args.aux_dim), dtype=np.float32)
        for i, p in enumerate(papers):
            if p["venue"]:
                venue_feat[i] = v_emb[v_row[p["venue"]]]

        orgs = sorted({p["org_of"][a] for p in train_p for a in p["clean"] if p["org_of"].get(a)})
        o_emb = pca_reduce(enc.encode("org", orgs, orgs), args.aux_dim, args.seed)
        o_row = {o: i for i, o in enumerate(orgs)}
        inst_sum = np.zeros((n_a, args.aux_dim), dtype=np.float32)
        inst_cnt = np.zeros((n_a, 1), dtype=np.float32)
        for p in train_p:
            for a in p["clean"]:
                o = p["org_of"].get(a)
                if o:
                    inst_sum[amap[a]] += o_emb[o_row[o]]
                    inst_cnt[amap[a]] += 1
        inst_feat = inst_sum / np.maximum(inst_cnt, 1)
        inst_t = torch.tensor(inst_feat, device=device)
    venue_t = torch.tensor(venue_feat, device=device) if use_vif else None
    text_t = torch.tensor(text, device=device)
    years_all = torch.tensor([[p["year"]] for p in papers], device=device)

    tr_rows = torch.tensor([pid_row[p["id"]] for p in train_p], device=device)
    P_text = text_t[tr_rows]
    P_text_n = F.normalize(P_text, dim=1)
    P_year = years_all[tr_rows]
    P_venue = venue_t[tr_rows] if use_vif else None
    n_p = len(train_p)

    # ---------------- graph + rule dictionaries ----------------
    print("Building bipartite graph and co-authorship rules ...")
    a_list, p_list = [], []
    co_cnt, co_last = Counter(), {}
    for j, p in enumerate(train_p):
        idx = [amap[a] for a in p["clean"]]
        for u in idx:
            a_list.append(u)
            p_list.append(j)
        for u in idx:
            for v in idx:
                if u != v:
                    co_cnt[(u, v)] += 1
                    if co_last.get((u, v), -1) < p["year"]:
                        co_last[(u, v)] = p["year"]
    a_idx = torch.tensor(a_list, device=device)
    p_idx = torch.tensor(p_list, device=device)
    global_freq = torch.bincount(a_idx, minlength=n_a).float()   # training papers per author
    t_ref = float(train_p[-1]["year"])
    edge_age = (t_ref - P_year.squeeze(1)[p_idx]).clamp(min=0)
    paper_cnt = torch.zeros(n_a, 1, device=device).index_add_(0, a_idx, torch.ones(len(a_list), 1, device=device)).clamp(min=1)

    # CSR: train paper -> its authors
    order = torch.argsort(p_idx)
    pa_authors = a_idx[order]
    pa_ptr = torch.zeros(n_p + 1, dtype=torch.long, device=device)
    pa_ptr[1:] = torch.cumsum(torch.bincount(p_idx, minlength=n_p), 0)

    def authors_of(paper_rows):
        chunks = [pa_authors[pa_ptr[r]:pa_ptr[r + 1]] for r in paper_rows.tolist()]
        return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.long, device=device)

    # CSR: author -> co-authors (with weight and last joint year)
    keys = sorted(co_cnt.keys())
    aa_src = torch.tensor([u for u, _ in keys], device=device)
    aa_dst = torch.tensor([v for _, v in keys], device=device)
    raw_w = torch.log1p(torch.tensor([co_cnt[k] for k in keys], dtype=torch.float32, device=device))
    aa_w = raw_w / raw_w.max() * 5.0
    aa_year = torch.tensor([co_last[k] for k in keys], dtype=torch.float32, device=device)
    aa_ptr = torch.zeros(n_a + 1, dtype=torch.long, device=device)
    aa_ptr[1:] = torch.cumsum(torch.bincount(aa_src, minlength=n_a), 0)

    print("Mining 2-hop hard negatives ...")
    adj = defaultdict(set)
    for u, v in keys:
        adj[u].add(v)
    HARD_POOL = 64
    hard_pool = torch.randint(0, n_a, (n_a, HARD_POOL), device=device)
    np_rng = np.random.default_rng(args.seed)
    for u in range(n_a):
        fof = set()
        for f in adj[u]:
            fof |= adj[f]
        fof -= adj[u]
        fof.discard(u)
        if fof:
            hard_pool[u] = torch.tensor(np_rng.choice(list(fof), HARD_POOL, replace=True), device=device)

    # ---------------- model ----------------
    time_enc = (LinearTimeEncoder() if args.linear_time else HarmonicTimeEncoder(args.time_dim)).to(device)
    in_p = text.shape[1] + time_enc.out_dim + (args.aux_dim if use_vif else 0)
    in_a = in_p + (args.aux_dim if use_vif else 0)
    model = DecayedLightGCN(in_p, in_a, args.hidden, dropout=0.2, learn_decay=not args.no_tew).to(device)
    logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.05), device=device))

    def paper_features(txt, yrs, ven):
        parts = [txt, time_enc(yrs)]
        if use_vif:
            parts.append(ven)
        return torch.cat(parts, dim=1)

    def author_features(xp):
        xa = torch.zeros(n_a, xp.size(1), device=device).index_add_(0, a_idx, xp[p_idx]) / paper_cnt
        return torch.cat([xa, inst_t], dim=1) if use_vif else xa

    params = list(model.parameters()) + list(time_enc.parameters()) + [logit_scale]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    pop_dist = global_freq / global_freq.sum()
    n_e = a_idx.numel()

    print(f"Training LightGCN: {n_p} papers, {n_a} authors, {n_e} edges, in_p={in_p}, in_a={in_a}")
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        xp = paper_features(P_text, P_year, P_venue)
        xa = author_features(xp)
        keep = torch.rand(n_e, device=device) > args.edge_dropout
        hp, ha, hp0, ha0 = model(xp, xa, a_idx[keep], p_idx[keep], edge_age[keep])

        bs = min(args.batch_size, n_e)
        b = torch.randperm(n_e, device=device)[:bs]
        bsrc, bdst = a_idx[b], p_idx[b]
        temp = logit_scale.exp().clamp(max=100)
        neg = torch.cat([
            torch.randint(0, n_a, (bs, 22), device=device),
            torch.multinomial(pop_dist, bs * 22, replacement=True).view(bs, 22),
            torch.gather(hard_pool[bsrc], 1, torch.randint(0, HARD_POOL, (bs, 20), device=device)),
        ], dim=1)
        pos = (hp[bdst] * ha[bsrc]).sum(1) * temp
        negs = (hp[bdst].unsqueeze(1) * ha[neg]).sum(2) * temp
        pos0 = (hp0[bdst] * ha0[bsrc]).sum(1) * temp
        negs0 = (hp0[bdst].unsqueeze(1) * ha0[neg]).sum(2) * temp
        loss = infonce(pos, negs) + 0.3 * infonce(pos0, negs0)
        loss.backward()
        opt.step()
        sched.step()
        if ep % 25 == 0 or ep == args.epochs - 1:
            lam = model.decay_rate().item()
            print(f"   epoch {ep:03d} | loss {loss.item():.4f} | lambda {lam:.4f} | {time.time() - t0:.0f}s")

    model.eval()
    with torch.no_grad():
        xp = paper_features(P_text, P_year, P_venue)
        _, A_emb, _, _ = model(xp, author_features(xp), a_idx, p_idx, edge_age)
    learned_lambda = model.decay_rate().item()

    # ---------------- experts ----------------
    def experts(p, valid, ctx):
        """Return raw score vectors (gnn, sem, rb), known author indices, fusion input features."""
        r = pid_row[p["id"]]
        q_txt = text_t[r:r + 1]
        q_year = years_all[r:r + 1]
        with torch.no_grad():
            xq = paper_features(q_txt, q_year, venue_t[r:r + 1] if use_vif else None)
            content = (F.normalize(model.proj_p(xq), dim=1) @ A_emb.t()).squeeze(0)
            sims = (F.normalize(q_txt, dim=1) @ P_text_n.t()).squeeze(0)
            known = torch.tensor([amap[a] for a in valid[:ctx]], dtype=torch.long, device=device)
            sw = SOCIAL_WEIGHT[ctx]
            if ctx == 0:
                boot = torch.unique(authors_of(torch.topk(sims, min(5, n_p)).indices))
                if boot.numel():
                    anchor = F.normalize(A_emb[boot].mean(0, keepdim=True), dim=1)
                    gnn = (1 - sw) * content + sw * (anchor @ A_emb.t()).squeeze(0)
                else:
                    gnn = content.clone()
                rb = global_freq.clone()
            else:
                anchor = F.normalize(A_emb[known].mean(0, keepdim=True), dim=1)
                gnn = (1 - sw) * content + sw * (anchor @ A_emb.t()).squeeze(0)
                rb = torch.zeros(n_a, device=device)
                yq = float(p["year"])
                for ki in known.tolist():
                    s, e = aa_ptr[ki].item(), aa_ptr[ki + 1].item()
                    if e > s:
                        dec = torch.exp(-RB_DECAY * (yq - aa_year[s:e]).clamp(min=0))
                        rb.index_add_(0, aa_dst[s:e], aa_w[s:e] * dec)
            top20 = torch.topk(sims, min(20, n_p))
            sem = torch.zeros(n_a, device=device)
            counts = (pa_ptr[top20.indices + 1] - pa_ptr[top20.indices])
            sem.index_add_(0, authors_of(top20.indices), top20.values.repeat_interleave(counts))
            if known.numel():
                gnn[known] = -float("inf")
                rb[known] = -float("inf")
                sem[known] = -float("inf")
            onehot = torch.zeros(3, device=device)
            onehot[ctx] = 5.0
            st = torch.tensor(score_stats(gnn) + score_stats(sem) + score_stats(rb), device=device)
            feats = torch.cat([xq.squeeze(0), onehot, st])
        return gnn, sem, rb, known, feats

    # ---------------- fusion training on the held-out validation split ----------------
    print(f"\nBuilding fusion training set from {min(len(val_q), args.fusion_papers)} validation papers "
          f"(their edges are NOT in the graph) ...")
    fus_q = rng.sample(val_q, min(len(val_q), args.fusion_papers))
    fus_items = []
    for p, valid in fus_q:
        for ctx in CONTEXT_LEVELS:
            g, s, b, known, feats = experts(p, valid, ctx)
            hidden = torch.tensor([amap[a] for a in valid[ctx:]], device=device)
            ng, ns, nb = minmax(g), minmax(s), minmax(b)
            kk = min(FUSION_CAND_K, n_a)
            cand = torch.unique(torch.cat([torch.topk(g, kk).indices, torch.topk(s, kk).indices,
                                           torch.topk(b, kk).indices, hidden]))
            if known.numel():
                cand = cand[~torch.isin(cand, known)]
            pos = torch.nonzero(torch.isin(cand, hidden)).squeeze(1)
            fus_items.append({"ctx": ctx, "feats": feats,
                              "stack": torch.stack([ng[cand], ns[cand], nb[cand]]), "pos": pos})

    def make_fusion():
        return nn.Sequential(nn.Linear(in_p + 3 + 6, 64), nn.ReLU(), nn.Linear(64, 3), nn.Softmax(dim=-1)).to(device)

    variants = {"REACH": (), "REACH_noGNN": ("G",), "REACH_noSEM": ("S",), "REACH_noRB": ("B",)}
    fusion_nets = {}
    for name, excl in variants.items():
        net = make_fusion()
        fopt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for ep in range(args.fusion_epochs):
            random.shuffle(fus_items)
            tot = 0.0
            for it in fus_items:
                w = net(it["feats"])
                comb = fuse(it["stack"], w, it["ctx"], excl)
                loss = -F.log_softmax(comb * 10.0, dim=0)[it["pos"]].mean()
                fopt.zero_grad()
                loss.backward()
                fopt.step()
                tot += loss.item()
            if ep % 10 == 0 or ep == args.fusion_epochs - 1:
                print(f"   fusion {name:<12} epoch {ep:02d} | loss {tot / max(len(fus_items), 1):.4f}")
        net.eval()
        fusion_nets[name] = (net, excl)

    # ---------------- static ensemble tuned on validation (per context) ----------------
    grid = torch.tensor([[i / 10, j / 10, (10 - i - j) / 10] for i in range(11) for j in range(11 - i)], device=device)
    static_w = {}
    for ctx in CONTEXT_LEVELS:
        items = [it for it in fus_items if it["ctx"] == ctx]
        totals = torch.zeros(grid.size(0), device=device)
        disc = 1.0 / torch.log2(torch.arange(2, 12, device=device, dtype=torch.float32))
        for it in items:
            g = grid.clone()
            if ctx == 0:
                g[:, 2] = 0
            g = g / (g.sum(1, keepdim=True) + 1e-8)
            sc = g @ it["stack"]                                   # (n_grid, |C|)
            top = torch.topk(sc, min(10, sc.size(1)), dim=1).indices
            hit = torch.isin(top, it["pos"]).float()
            idcg = disc[: min(len(it["pos"]), 10)].sum()
            totals += (hit * disc[: hit.size(1)]).sum(1) / idcg
        best = grid[int(torch.argmax(totals))]
        if ctx == 0:
            best = best.clone()
            best[2] = 0
        static_w[ctx] = (best / best.sum()).tolist()
        print(f"   tuned static weights C={ctx} [GNN, SEM, RB] = {[round(x, 2) for x in static_w[ctx]]}")

    # ---------------- test ----------------
    print(f"\nEvaluating on {len(test_q)} test papers ...")
    maxk = max(K_EVAL)
    per_q = {m: {c: {k: ([], []) for k in K_EVAL} for c in CONTEXT_LEVELS} for m in METHODS}
    gate_w = {c: [] for c in CONTEXT_LEVELS}
    fixed_w = torch.tensor([0.4, 0.4, 0.2], device=device)
    t0 = time.time()
    for qi, (p, valid) in enumerate(test_q):
        for ctx in CONTEXT_LEVELS:
            g, s, b, known, feats = experts(p, valid, ctx)
            hidden = {amap[a] for a in valid[ctx:]}
            with torch.no_grad():
                stack = torch.stack([minmax(g), minmax(s), minmax(b)])
                scores = {"GNN": g, "SEM": s, "RB": b,
                          "WHO_FIXED": fuse(stack, fixed_w, ctx),
                          "STATIC_TUNED": fuse(stack, torch.tensor(static_w[ctx], device=device), ctx)}
                for name, (net, excl) in fusion_nets.items():
                    w = net(feats)
                    if name == "REACH":
                        wk = w.clone()
                        if ctx == 0:
                            wk[2] = 0
                        gate_w[ctx].append((wk / wk.sum()).tolist())
                    scores[name] = fuse(stack, w, ctx, excl)
                for name in ("WHO_FIXED", "STATIC_TUNED", *fusion_nets):
                    if known.numel():
                        scores[name][known] = -float("inf")
            for name, sc in scores.items():
                top = torch.topk(sc, min(maxk, n_a)).indices.tolist()
                for K, (nd, rc) in ndcg_recall_prefix(top, hidden, K_EVAL).items():
                    per_q[name][ctx][K][0].append(nd)
                    per_q[name][ctx][K][1].append(rc)
        if (qi + 1) % 2000 == 0:
            print(f"   {qi + 1}/{len(test_q)} test papers | {time.time() - t0:.0f}s")

    # ---------------- save ----------------
    rows = []
    for m in METHODS:
        for c in CONTEXT_LEVELS:
            for K in K_EVAL:
                nd, rc = per_q[m][c][K]
                rows.append((m, c, K, float(np.mean(nd)) if nd else float("nan"),
                             float(np.mean(rc)) if rc else float("nan"), len(nd)))
    with open(os.path.join(run_dir, "metrics.csv"), "w") as f:
        f.write("method,context,K,ndcg,recall,n_queries\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.6f},{r[4]:.6f},{r[5]}\n")
    np.savez_compressed(os.path.join(run_dir, "per_query_ndcg.npz"),
                        **{f"{m}_C{c}_K{K}": np.array(per_q[m][c][K][0], dtype=np.float32)
                           for m in METHODS for c in CONTEXT_LEVELS for K in K_EVAL})

    # Paired bootstrap: REACH vs each baseline on per-query NDCG@K.
    boot_rng = np.random.default_rng(args.seed)
    sig = []
    for c in CONTEXT_LEVELS:
        for K in (5, 10, 20):
            r = np.array(per_q["REACH"][c][K][0])
            if len(r) == 0:
                continue
            for base in ("GNN", "RB", "SEM", "WHO_FIXED", "STATIC_TUNED"):
                d = r - np.array(per_q[base][c][K][0])
                bm = np.concatenate([d[boot_rng.integers(0, len(d), size=(min(200, args.bootstrap - i), len(d)))].mean(1)
                                     for i in range(0, args.bootstrap, 200)])
                lo, hi = np.percentile(bm, [2.5, 97.5])
                p_two = min(1.0, 2 * min((bm <= 0).mean(), (bm >= 0).mean()))
                sig.append((c, K, base, d.mean(), lo, hi, p_two))
    with open(os.path.join(run_dir, "significance.csv"), "w") as f:
        f.write("context,K,baseline,reach_minus_baseline_ndcg,ci95_low,ci95_high,p_value\n")
        for s_ in sig:
            f.write(f"{s_[0]},{s_[1]},{s_[2]},{s_[3]:.6f},{s_[4]:.6f},{s_[5]:.6f},{s_[6]:.4f}\n")

    mean_gate = {c: (np.mean(gate_w[c], axis=0).round(4).tolist() if gate_w[c] else None) for c in CONTEXT_LEVELS}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "learned_lambda": learned_lambda, "t_ref": t_ref,
                   "static_tuned_weights_[GNN,SEM,RB]": static_w,
                   "mean_reach_gate_weights_[GNN,SEM,RB]": mean_gate}, f, indent=2)

    # ---------------- print ----------------
    print(f"\nLearned lambda: {learned_lambda:.4f} | mean REACH gate [GNN, SEM, RB]: {mean_gate}")
    show = ["GNN", "RB", "SEM", "WHO_FIXED", "STATIC_TUNED", "REACH"]
    print("\nNDCG@K  " + " ".join(f"{m:>12}" for m in show))
    for c in CONTEXT_LEVELS:
        for K in (5, 10, 20, 100):
            vals = [np.mean(per_q[m][c][K][0]) for m in show]
            print(f"C={c} K={K:<3} " + " ".join(f"{v:>12.4f}" for v in vals))
    print("\nREACH vs baselines (NDCG, paired bootstrap):")
    for s_ in sig:
        print(f"   C={s_[0]} K={s_[1]:<2} vs {s_[2]:<12} diff {s_[3]:+.4f} [{s_[4]:+.4f}, {s_[5]:+.4f}] p={s_[6]:.4f}")
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()

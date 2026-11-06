# WhoIsWho Author Recommendation

**Predicting the missing authors of a scientific paper.** Given a paper's title, abstract and year, plus zero, one or two of its known authors, the model ranks every researcher in the network by how likely they are to be a co-author.

The final model, **REACH**, combines three signals with a small learned gating network:

| Engine | What it scores | Implementation |
|---|---|---|
| **GNN** | Structural fit in the author–paper graph | 2-layer bipartite **LightGCN** with a learned temporal edge decay, over SPECTER2 paper embeddings + harmonic time encoding + venue / institution features, trained with InfoNCE and 2-hop hard negatives |
| **SEM** | Topical fit | Cosine similarity of the paper's **SPECTER2** embedding to the 20 most similar training papers, aggregated to their authors |
| **RB** | Social fit | Co-authorship rules: weighted co-occurrence with the known authors, with exponential time decay since their last joint paper |

A fusion MLP reads the paper embedding, the context level (how many authors are known) and summary statistics of each engine's scores, then outputs per-paper weights for the three engines. With no known authors, the rule engine is switched off automatically.

The project developed over about 10 months, from TF-IDF baselines through transformer-embedding GNNs, clustering, GATv2 and GraphSAGE to the final fusion model. All of those stages are kept in this repo, in order.

---

## Results

### v1 (the numbers in the August 2026 draft, `reach_lightgcn_fusion.py`)

NDCG@K on the temporal test split:

| Known authors | K | GNN | RB | SEM | REACH |
|---|---|---|---|---|---|
| 0 | 5 | 0.090 | 0.013 | **0.173** | 0.162 |
| 0 | 10 | 0.100 | 0.016 | **0.185** | 0.173 |
| 1 | 5 | 0.171 | 0.271 | 0.162 | **0.278** |
| 1 | 10 | 0.182 | 0.287 | 0.176 | **0.295** |
| 2 | 5 | 0.196 | 0.297 | 0.150 | **0.299** |
| 2 | 10 | 0.209 | 0.316 | 0.164 | **0.318** |

In v1 the fusion layer was trained on papers whose edges were already in the graph, so it over-trusted the graph experts. It lost to plain semantic search with no known authors, and only just beat the co-authorship rules once authors were known.

### v2 (`reach.py`)

v2 fixes that leak and adds the missing components and baselines:

- **Leak-free fusion:** chronological 70 / 10 / 20 split. The fusion layer is trained on the 10% validation slice, whose papers are not in the graph.
- **Venue and institution features (VIF):** SPECTER2 embeddings of venue names and each author's affiliations, reduced with PCA to 128 dimensions.
- **Learned temporal decay (TEW):** edge weights `exp(-λ (T − year))` in the LightGCN adjacency, with λ learned end to end.
- **Proper k-core pruning:** counts are recomputed on the pruned author lists each iteration.
- **Baselines:** the fixed 0.4/0.4/0.2 WhoIsWho ensemble, plus a static ensemble grid-searched per context level on validation.
- **Ablations:** fusion without the GNN, SEM or RB expert, plus `--no-vif`, `--no-tew` and `--linear-time`.
- **Reporting:** multiple seeds, paired-bootstrap 95% CIs and p-values, dataset statistics (Table I), learned λ and the average gate weights per context.

To reproduce the paper tables:

```bash
python experiments/07_reach_final/run_experiments.py --seeds 0 1 2      # full model + 3 ablations
# → results/runs/<config>_seed<N>/  and  results/summary.md (Tables I, III, IV + significance)
```

## Repository structure

```
whoiswho-author-recommendation/
├── data/                         # put pid_to_info_all.json here (not tracked; see data/README.md)
├── experiments/
│   ├── 01_tfidf_baselines/       # Aug–Oct 2025: TF-IDF nearest-paper author lookup vs random baseline
│   ├── 02_transformer_gnn_hpo/   # Oct 2025: SciBERT / SPECTER embeddings + manual hetero GCN, Optuna HPO
│   ├── 03_clustering_and_rules/  # Nov–Dec 2025: GMM/EM author clustering, title-topic clusters, association rules
│   ├── 04_hetero_gnn_recommender/# Jan–Feb 2026: hetero GNN recommender, k-core subsets, NDCG evaluation
│   ├── 05_gat_and_graphsage/     # Feb–May 2026: GATv2 + jumping knowledge, context masks, GraphSAGE + Borda fusion
│   ├── 06_ablations/             # Apr–May 2026: author-only graph vs author+paper hetero graph
│   └── 07_reach_final/           # REACH: reach.py (v2), run_experiments.py, aggregate_runs.py, v1 script
├── analysis/
│   ├── dataset_stats.py          # papers / authors / venues / institutions per period (2010–15, 2015–20, 2010–20)
│   └── plot_poster_results.py    # regenerates the poster results figure
├── utils/
│   └── check_intel_xpu.py        # checks Intel Arc GPU (XPU) availability
├── results/
│   ├── grid_search_results.csv   # fusion hyperparameter grid search
│   ├── clusters/{hard,soft}/KEYWORDS.txt   # top keywords per EM topic cluster (k=15)
│   └── figures/                  # charts produced by the experiments
├── requirements.txt
└── README.md
```

### What each script does

| Script | Original name | Description |
|---|---|---|
| `01_tfidf_baselines/tfidf_baseline.py` | `concept.py` | TF-IDF (title + abstract) nearest papers → candidate authors, hit rate vs a random baseline |
| `01_tfidf_baselines/tfidf_baseline_keywords.py` | `similarity.py` | Same, with keywords added to the text and papers sorted by year |
| `01_tfidf_baselines/tfidf_kmeans_ndcg.py` | `realtest.py` | Adds NDCG scoring and MiniBatch K-Means clustering |
| `02_.../scibert_gnn_optuna.py` | `para-test.py` | SciBERT embeddings + hand-written heterogeneous GCN, Optuna tuning (P/R/F1) |
| `02_.../scibert_gnn_optuna_ndcg.py` | `hyper-parameter-testing.py` | Same, with NDCG and co-author pair edges |
| `02_.../specter_gnn_cached_embeddings.py` | `scibert-vector-embeddings.py` | Switchable SPECTER/SciBERT encoder with an embedding cache, final training and evaluation |
| `02_.../specter_gnn_final_params.py` | `final-parameter-tests.py` | Final fixed-parameter run of the SPECTER GCN |
| `03_.../gmm_author_clustering.py` | `expectation-maximisation.py` | GCN author embeddings → GMM (EM) clustering with BIC model selection and cluster analysis |
| `03_.../gmm_clustering_association_rules.py` | `association-maximisation-test.py` | Same, plus Apriori co-authorship association rules (mlxtend) |
| `03_.../em_title_clusters_n1.py` / `_n3.py` | `specter-title-cluster-n1/n3.py` | SPECTER title topic clusters (hard and soft EM, k=15), keyword extraction, rule mining; writes `em_results_hard/` and `em_results_soft/` |
| `04_.../hetero_gnn_minilm.py` | `new-model.py` | Hetero GNN recommender with MiniLM embeddings, scaling plot (`performance_scaling.png`) |
| `04_.../hetero_gnn_kcore.py` | `other-model.py` | mpnet embeddings, k-core subsets, accuracy vs masked authors (`final_report_chart.png`) |
| `04_.../hetero_gnn_kcore_ndcg.py` | `february.py` | Same family, evaluated with NDCG/recall |
| `05_.../gatv2_jk.py` | `16-february.py` | Multi-head GATv2 bipartite model with jumping knowledge; saves `best_gat_model.pt` and `final_author_embeddings.pt` |
| `05_.../gatv2_jk_context_masks.py` | `9-march.py` | GATv2-JK evaluated at mask levels 0/1/2/3/ALL |
| `05_.../graphsage_borda_full_dataset.py` | `11-march.py` | Bipartite GraphSAGE on the full dataset with contextual Borda fusion of GNN + SEM + rules |
| `06_ablations/author_only_gat.py` | `author-only.py` | Ablation: GAT on the author–author graph only |
| `06_ablations/author_paper_hetero_gat.py` | `author-paper.py` | Ablation: hetero GAT with author→paper, paper→author and author→author edges |
| `07_reach_final/reach_lightgcn_fusion.py` | `10-march.py` | REACH **v1** (draft results), with ablations (no-SEM, no-RB), a static 0.4/0.4/0.2 baseline and per-component timing |
| `07_reach_final/reach.py` | – | **REACH v2**: leak-free fusion, VIF, learned decay, tuned static baseline, ablation flags, significance tests |
| `07_reach_final/run_experiments.py` | – | Runs seeds × {full, noVIF, noTEW, linearTE}, then aggregates |
| `07_reach_final/aggregate_runs.py` | – | Builds `results/summary.md` (mean ± std tables) from all runs |

---

## Getting started

### 1. Clone and set up an environment

```bash
git clone https://github.com/dunhuahua/whoiswho-author-recommendation.git
cd whoiswho-author-recommendation

python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

**GPU (recommended for stages 05–07):** install the PyTorch build that matches your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/) *before* running `pip install -r requirements.txt`. Scripts use CUDA when it's available and fall back to CPU otherwise. The ablation scripts also support Intel Arc GPUs via `intel-extension-for-pytorch` (see `utils/check_intel_xpu.py`).

### 2. Get the data

The dataset is `pid_to_info_all.json` (~840 MB) from the **WhoIsWho** benchmark (AMiner / THUDM). It's too large for git. Download instructions are in [`data/README.md`](data/README.md). Put the file here:

```
data/pid_to_info_all.json
```

### 3. Run

**Run every script from the repository root.** Data paths are relative to the root, and embedding caches (`*.pkl`), model checkpoints (`*.pt`) and figures are written there too (git ignores them).

```bash
# 1-minute smoke test on synthetic data (CPU is fine, no dataset needed)
python experiments/07_reach_final/reach.py --synthetic 3000 --epochs 20 --fusion-papers 200

# Final REACH v2 model, one seed (GPU strongly recommended)
python experiments/07_reach_final/reach.py --seed 0

# Faster iteration on a subset
python experiments/07_reach_final/reach.py --seed 0 --max-papers 60000 --max-test-papers 5000

# All seeds and ablations, then the summary tables
python experiments/07_reach_final/run_experiments.py --seeds 0 1 2

# Dataset statistics, earlier stages
python analysis/dataset_stats.py
python experiments/05_gat_and_graphsage/gatv2_jk.py
```

On the first run, SPECTER2 / mpnet / SciBERT weights are downloaded (`reach.py` caches its embeddings as `cache/*.npy`) from Hugging Face and paper embeddings are computed and cached as `*.pkl`. Later runs load the cache. Several stage 02–04 scripts generate synthetic data automatically if `data/pid_to_info_all.json` is missing, which is useful as a quick smoke test.

Hyperparameters (dataset size `SIZES_TO_TEST`, `K_VALUES`, `CONTEXT_LEVELS`, learning rate, dropout and so on) are constants at the top of each script.

### Hardware notes

- Embedding the full dataset with SPECTER2 takes a long time on CPU. A GPU with ≥ 8 GB VRAM is recommended.
- The final model uses a "memory-safe" sparse LightGCN (`index_add_` message passing, batch size 16384), so it doesn't need PyTorch Geometric.
- Loading the full JSON needs roughly 4–6 GB of RAM.

---

## Method details

- **Preprocessing:** iterative **k-core pruning**. Authors must appear on ≥ 3 papers and papers must keep ≥ 3 such authors, repeated until the graph converges, which removes noise from one-off author IDs.
- **Text encoder:** `allenai/specter2_base` (sentence-transformers) on title + abstract. Earlier stages compare SciBERT, SPECTER, all-mpnet-base-v2 and all-MiniLM-L6-v2.
- **Time:** learnable sinusoidal (harmonic) year encoding concatenated to paper features. Graph edges are weighted by `exp(-λ · age)` with λ learned (v2). In the rule engine, co-authorship strength decays as `exp(-0.138 · Δyears)`.
- **Training:** InfoNCE loss with friends-of-friends hard negatives and edge dropout for regularisation.
- **Evaluation:** NDCG@K and Recall@K at 0, 1 and 2 known authors, on the newest 20% of papers. Candidates are authors seen in the training graph, and every test paper keeps ≥ 3 in-graph authors.
- **Author identity caveat:** if a record has no author `id`, the author's **name string** is used as the ID. WhoIsWho exists precisely because names are ambiguous, so check which field your copy of the data provides.

## Acknowledgements

- Dataset: Chen et al., *Web-Scale Academic Name Disambiguation: the WhoIsWho Benchmark, Leaderboard, and Toolkit*, KDD 2023. [arXiv:2302.11848](https://arxiv.org/abs/2302.11848) · [THUDM/WhoIsWho](https://github.com/THUDM/WhoIsWho)
- Embeddings: [SPECTER2](https://huggingface.co/allenai/specter2_base) (AllenAI), [sentence-transformers](https://www.sbert.net/)

## Author

**Duncan Alexander**, BSc Data Science, University of Sheffield · [github.com/dunhuahua](https://github.com/dunhuahua)

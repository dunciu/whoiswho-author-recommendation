# Data

The scripts expect one file in this folder:

```
data/pid_to_info_all.json     (~840 MB, not tracked by git)
```

## Where it comes from

`pid_to_info_all.json` is the paper-metadata file of the **WhoIsWho** author name disambiguation benchmark from AMiner / Tsinghua (THUDM). It maps each paper ID to its metadata:

```json
{
  "9PgiwDo7": {
    "id": "9PgiwDo7",
    "title": "...",
    "abstract": "...",
    "keywords": ["..."],
    "authors": [{"name": "...", "org": "...", "id": "..."}],
    "venue": "...",
    "year": 2019
  }
}
```

Download options:

1. **WhoIsWho toolkit / benchmark page:** https://github.com/THUDM/WhoIsWho. Follow the dataset download links in that README (WhoIsWho v3 / IND task data).
2. **OAG-Bench / KDD Cup 2024 (IND task)**, which redistributes the same `pid_to_info_all.json`.

After downloading, put the file in this folder so that `data/pid_to_info_all.json` exists.

## Generated files

On the first run, the scripts cache sentence embeddings as `*.pkl` (for example `allenai_specter2_base_embeddings_20000.pkl`) and save checkpoints as `*.pt` in the repo root. Git ignores them, and they're rebuilt automatically when missing.

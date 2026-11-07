"""Shared loaders, run-file I/O, timing and seeding for the T1 spike scripts."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent   # project root
DATA = HERE / "data"          # CS410 MP collections (MeTA line corpora)
DATA_BEIR = HERE / "data_beir"               # BEIR collections (python -m retrieval_eval.data.fetch_beir)
RUNS = HERE / "runs"
TIMING = HERE / "timing"
BEIR_SETS = ["scifact", "nfcorpus", "fiqa", "trec-covid"]
MP_SETS = ["cranfield", "new_faculty", "apnews"]
DATASETS = BEIR_SETS  # default: the semantic-search evaluation sets
TOPK = 100   # depth of every run file written
SEED = 0     # global seed for random / numpy / torch

# MeTA line corpora: doc id = 0-based line number (verified in Step 0).
# Query ids: line index + start (verified against qrels in Step 0).
QUERY_ID_START = {"cranfield": 1, "new_faculty": 1, "apnews": 0}
# Expected counts from Step 0: docs, queries, qrels rows, judged qids, qids with a relevant doc.
EXPECTED = {
    "cranfield": (1400, 225, 1612, 225, 225),
    "new_faculty": (7014, 2234, 42834, 2234, 2203),
    "apnews": (164465, 100, 10892, 99, 99),
    # BEIR test split (Step 0, step0_probe_beir.txt; matches the BEIR paper's table)
    "scifact": (5183, 300, 339, 300, 300),
    "nfcorpus": (3633, 323, 12334, 323, 323),
    "fiqa": (57638, 648, 1706, 648, 648),
    "trec-covid": (171332, 50, 66336, 50, 50),
}


def _lines(path):
    """Read a MeTA line corpus as a list of lines (one doc/query per line), dropping the trailing empty line."""
    # Split on "\n" only: str.splitlines() also breaks on \x0b, \x1c,  etc.
    # which would shift line-number doc ids.
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def load_beir(ds):
    """BEIR format: corpus.jsonl (_id, title, text), queries.jsonl, qrels/test.tsv (with header).
    Doc text = title + " " + text (BEIR convention). Only test-split queries are used.
    Negative grades (2 rows in trec-covid) are clamped to 0 (judged non-relevant)."""
    base = DATA_BEIR / ds
    corpus = {}
    for line in open(base / "corpus.jsonl", encoding="utf-8"):
        o = json.loads(line)
        corpus[o["_id"]] = f"{o.get('title', '')} {o.get('text', '')}".strip()
    qrels = defaultdict(dict)
    rows = 0
    with open(base / "qrels" / "test.tsv", encoding="utf-8") as f:
        next(f)  # header: query-id corpus-id score
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) == 3:  # skip blank/malformed rows
                qrels[p[0]][p[1]] = max(0, int(p[2]))
                rows += 1
    qrels = dict(qrels)
    queries = {}
    for line in open(base / "queries.jsonl", encoding="utf-8"):
        o = json.loads(line)
        if o["_id"] in qrels:  # queries.jsonl holds all splits; keep only test qids
            queries[o["_id"]] = o["text"].strip()
    return corpus, queries, qrels, rows


def load_beir_fields(ds):
    """{doc_id: (title, text)} for multi-field indexing (BEIR's Anserini BM25 setup)."""
    out = {}
    for line in open(DATA_BEIR / ds / "corpus.jsonl", encoding="utf-8"):
        o = json.loads(line)
        out[o["_id"]] = (o.get("title", "") or "", o.get("text", "") or "")
    return out


def load(ds):
    """Load dataset `ds` and validate it against EXPECTED.

    Returns (corpus {doc_id: text}, queries {qid: text}, qrels {qid: {doc_id: grade}}), all ids as str.
    BEIR sets use load_beir; MP sets are MeTA line corpora where the doc id is the line number.
    """
    if ds in BEIR_SETS:
        corpus, queries, qrels, rows = load_beir(ds)
        return _check(ds, corpus, queries, qrels, rows)
    base = DATA / ds
    docs = _lines(base / f"{ds}.dat")
    corpus = {str(i): t for i, t in enumerate(docs)}
    start = QUERY_ID_START[ds]
    queries = {str(i + start): t.strip() for i, t in enumerate(_lines(base / f"{ds}-queries.txt"))}
    # qrels lines: "qid docid grade" (whitespace separated)
    qrels = defaultdict(dict)
    rows = 0
    for line in _lines(base / f"{ds}-qrels.txt"):
        p = line.split()
        if p:
            qrels[p[0]][p[1]] = int(p[2])
            rows += 1
    qrels = dict(qrels)
    return _check(ds, corpus, queries, qrels, rows)


def _check(ds, corpus, queries, qrels, rows):
    """Assert counts match the Step 0 probe and every qrels id resolves; returns (corpus, queries, qrels)."""
    n_rel = sum(1 for v in qrels.values() if any(g > 0 for g in v.values()))
    got = (len(corpus), len(queries), rows, len(qrels), n_rel)
    assert got == EXPECTED[ds], f"{ds}: counts {got} != Step 0 {EXPECTED[ds]}"
    assert all(d in corpus for v in qrels.values() for d in v), f"{ds}: unresolved qrels doc id"
    assert all(q in queries for q in qrels), f"{ds}: qrels qid without query"
    return corpus, queries, qrels


def write_run(ds, system, run):
    """Write runs/{ds}.{system}.trec in TREC format.

    run: {qid: [docid, ...]} in rank order. Score = TOPK + 1 - rank keeps order exact under trec_eval
    (trec_eval re-sorts by score, so original retrieval scores could reorder ties).
    """
    RUNS.mkdir(exist_ok=True)
    with open(RUNS / f"{ds}.{system}.trec", "w") as f:
        for qid, docs in run.items():
            for r, d in enumerate(docs, 1):
                f.write(f"{qid} Q0 {d} {r} {TOPK + 1 - r} {system}\n")


def read_run(ds, system):
    """Read a TREC run file back into {qid: [docid, ...]} ordered by the rank column."""
    run = defaultdict(list)
    for line in open(RUNS / f"{ds}.{system}.trec"):
        qid, _, d, r, _, _ = line.split()
        run[qid].append((int(r), d))
    return {q: [d for _, d in sorted(v)] for q, v in run.items()}


def write_timing(ds, system, info):
    """Save a system's timing/metadata dict as timing/{ds}.{system}.json."""
    TIMING.mkdir(exist_ok=True)
    (TIMING / f"{ds}.{system}.json").write_text(json.dumps(info, indent=2))


def read_timing(ds, system):
    """Load the dict written by write_timing."""
    return json.loads((TIMING / f"{ds}.{system}.json").read_text())


def qid_key(q):
    """Sort key for mixed numeric/string query ids (e.g. nfcorpus PLAIN-123).
    Numeric ids sort first, numerically; string ids follow, lexicographically."""
    return (0, int(q), "") if q.isdigit() else (1, 0, q)


def hf_revision(model_id):
    """Commit hash of the locally cached Hugging Face model (for reproducibility logs), or "unknown"."""
    ref = Path.home() / ".cache/huggingface/hub" / f"models--{model_id.replace('/', '--')}" / "refs/main"
    return ref.read_text().strip() if ref.exists() else "unknown"


def seed_all():
    """Seed Python, NumPy and (if installed) torch RNGs with SEED."""
    random.seed(SEED)
    np.random.seed(SEED)
    try:
        import torch
        torch.manual_seed(SEED)
    except ImportError:
        pass


def device():
    """Best available torch device: Apple MPS, then CUDA, then CPU."""
    import torch
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def cli(desc, extra=None):
    """Parse the shared --datasets flag; `extra(ap)` may add script-specific arguments. Returns argparse Namespace."""
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=BEIR_SETS + MP_SETS)
    if extra:
        extra(ap)
    return ap.parse_args()

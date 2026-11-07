"""Step 0 probe for the BEIR test splits: print counts, grade distribution, id-resolution rates
and a few sample query/relevant-doc pairs per dataset (output saved as step0_probe_beir.txt).
The counts printed here are the source of EXPECTED in retrieval_eval/common.py.
"""
import json, sys, collections, random
from pathlib import Path
B = Path(__file__).resolve().parents[2] / "data_beir"
random.seed(0)  # fixed sample of example queries
for ds in ["scifact", "nfcorpus", "fiqa", "trec-covid"]:
    d = B / ds
    corpus = {}
    keys = collections.Counter()  # which fields appear in corpus records
    for line in open(d / "corpus.jsonl"):
        o = json.loads(line); keys.update(o.keys()); corpus[o["_id"]] = o
    queries = {}
    for line in open(d / "queries.jsonl"):
        o = json.loads(line); queries[o["_id"]] = o
    rows = [l.rstrip("\n").split("\t") for l in open(d / "qrels/test.tsv")]
    header, rows = rows[0], [r for r in rows[1:] if r and r[0]]  # drop header and blank lines
    qrels = collections.defaultdict(dict)
    for q, doc, g in rows: qrels[q][doc] = int(g)
    grades = collections.Counter(int(g) for _, _, g in rows)
    n_rel = sum(1 for v in qrels.values() if any(g > 0 for g in v.values()))
    # Fraction of qrels doc ids / qids that exist in the corpus / query file (should both be 1.0).
    res = sum(1 for _, doc, _ in rows if doc in corpus) / len(rows)
    qres = sum(1 for q in qrels if q in queries) / len(qrels)
    empty = sum(1 for o in corpus.values() if not (o.get("title", "") + o.get("text", "")).strip())
    print(f"\n===== {ds}: docs={len(corpus)} (empty={empty}) all_queries={len(queries)} test_qids={len(qrels)} "
          f"test_qids_with_rel={n_rel} qrels_rows={len(rows)} header={header} grades={dict(sorted(grades.items()))} "
          f"doc_resolve={res:.4f} qid_resolve={qres:.4f} corpus_keys={dict(keys)}")
    # Spot-check: first doc, then two random test queries with one relevant doc each.
    print("  head corpus:", json.dumps(next(iter(corpus.values())))[:250])
    for q in random.sample(sorted(qrels), 2):
        rel = [x for x, g in qrels[q].items() if g > 0][:1]
        print(f"  Q {q}: {queries[q]['text'][:120]!r}")
        for x in rel:
            print(f"    rel doc {x} (g={qrels[q][x]}): {(corpus[x].get('title','') + ' | ' + corpus[x]['text'])[:160]!r}")

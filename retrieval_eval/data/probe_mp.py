"""Step 0 probe for the CS410 MP collections (MeTA line corpora): print counts and test whether
qrels doc ids are 0- or 1-based line numbers, then show sample queries with relevant docs so the
mapping can be checked by eye (output saved as step0_probe.txt). Results fixed QUERY_ID_START
and the doc-id convention in retrieval_eval/common.py.
"""
import random, sys, collections
D = "/Users/cwilkins/Projects/school/cs410/tech_review/data"
random.seed(0)  # fixed sample of example queries
for ds in ["cranfield", "new_faculty", "apnews"]:
    # Split on "\n" only (see common._lines) so line numbers match MeTA doc ids.
    docs = open(f"{D}/{ds}/{ds}.dat", encoding="utf-8", errors="replace").read().split("\n")
    if docs and docs[-1] == "": docs.pop()
    qs = open(f"{D}/{ds}/{ds}-queries.txt", encoding="utf-8").read().split("\n")
    if qs and qs[-1] == "": qs.pop()
    qrels = collections.defaultdict(dict)
    for line in open(f"{D}/{ds}/{ds}-qrels.txt"):
        p = line.split()
        if p: qrels[int(p[0])][int(p[1])] = int(p[2])
    rows = sum(len(v) for v in qrels.values())
    print(f"\n===== {ds}: docs={len(docs)} queries={len(qs)} qrels_rows={rows} qids={len(qrels)} qid_range=[{min(qrels)},{max(qrels)}]")
    # Resolution rate if doc ids are line numbers with offset 0 vs 1 (the correct one should be 1.0).
    for off in (0, 1):
        res = sum(1 for v in qrels.values() for d in v if 0 <= d - off < len(docs))
        print(f"  doc offset {off}: resolve {res/rows:.4f}")
    qoff = 0 if min(qrels) == 0 else 1  # infer query-id start from the smallest qid
    print(f"  queries with >=1 rel (grade>0): {sum(1 for v in qrels.values() if any(g>0 for g in v.values()))}")
    # Print each sample's relevant docs under both offsets so the right alignment is obvious.
    for qid in random.sample(sorted(qrels), 3):
        print(f"\n  QID {qid} -> query line {qid-qoff}: {qs[qid-qoff][:150]!r}")
        rel = [d for d, g in sorted(qrels[qid].items(), key=lambda x: -x[1]) if g > 0][:2]
        for d in rel:
            for off in (0, 1):
                print(f"    doc {d} (g={qrels[qid][d]}) off{off}: {docs[d-off][:160]!r}")

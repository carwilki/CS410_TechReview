"""S3 Hybrid: Reciprocal Rank Fusion of S1 (BM25) and S2 (Dense) top-100 lists.

Source: Cormack, Clarke & Buettcher (2009); PyTerrier docs (RRFusion) - score(d) =
sum over lists of 1/(k + rank(d)), k = 60, ranks starting at 1. Implemented by hand
rather than via PyTerrier. Keeps top 100. Reads runs/{ds}.BM25.trec and runs/{ds}.Dense.trec.
"""
import time
from collections import defaultdict

from retrieval_eval.common import TOPK, cli, qid_key, read_run, read_timing, write_run, write_timing

RRF_K = 60


def rrf(lists):
    """Fuse ranked doc-id lists with Reciprocal Rank Fusion; return the top TOPK ids by fused score.

    Only ranks are used, so BM25 and cosine scores need no normalization. A doc missing
    from a list simply gets no contribution from it.
    """
    score = defaultdict(float)
    for lst in lists:
        for r, d in enumerate(lst, 1):
            score[d] += 1.0 / (RRF_K + r)
    # Stable tie-break: first-seen order (BM25 list first) via sorted() stability.
    return [d for d, _ in sorted(score.items(), key=lambda x: -x[1])][:TOPK]


def main():
    """Fuse the existing BM25 and Dense runs per query and write the Hybrid run and timing."""
    args = cli(__doc__)
    for ds in args.datasets:
        bm, de = read_run(ds, "BM25"), read_run(ds, "Dense")
        run, times = {}, []
        # Union of qids: BM25 may have no entry for queries with no matching terms.
        for qid in sorted(set(bm) | set(de), key=qid_key):
            t = time.perf_counter()
            run[qid] = rrf([bm.get(qid, []), de.get(qid, [])])
            times.append(time.perf_counter() - t)
        write_run(ds, "Hybrid", run)
        fuse_ms = 1000 * sum(times) / len(times)
        # End-to-end latency assumes both retrievers run sequentially, then fusion.
        total = read_timing(ds, "BM25")["mean_ms"] + read_timing(ds, "Dense")["mean_ms"] + fuse_ms
        write_timing(ds, "Hybrid", {"system": "Hybrid", "fusion_ms": fuse_ms, "mean_ms": total,
                                    "n_queries": len(times), "rrf_k": RRF_K,
                                    "note": "mean_ms = BM25 + Dense + fusion (sequential end-to-end)"})
        print(f"[Hybrid] {ds}: fusion {fuse_ms:.3f} ms/q, end-to-end {total:.2f} ms/q", flush=True)


if __name__ == "__main__":
    main()

"""Cross-encoder reranking of a first-stage shortlist.

Default (S4 Rerank): rescore the Hybrid top-50; positions 51-100 keep Hybrid order.
Reproduction (BM25+CE): --source BM25-BEIR --depth 100 --name BM25+CE, i.e. BEIR's setup:
"BM25 + CE reranks the top-100 retrieved hits from a first-stage BM25 (Anserini) model"
with "a 6-layer, 384-h MiniLM" cross-encoder (BEIR Sec. 4(v); Table 6 links
cross-encoder/ms-marco-MiniLM-L-6-v2). Published BM25+CE nDCG@10 (BEIR Table 2):
SciFact 0.688, NFCorpus 0.350, FiQA 0.347, TREC-COVID 0.757.

Sources: BEIR (Thakur et al., 2021) for rerank-the-shortlist; RankGPT (Sun et al., 2023,
Sec. 6.5 / Table 5) found an LLM reranker "highly sensitive to the initial passage order"
(TREC-DL19, gpt-3.5-turbo), which motivates reranking the stronger Hybrid list in S4.
Document text = title + " " + text, as in BEIR's reranker (beir/reranking/rerank.py).
"""
import time

import numpy as np
from sentence_transformers import CrossEncoder
from transformers.utils import logging as hf_logging

from retrieval_eval.common import cli, device, hf_revision, load, read_run, read_timing, seed_all, write_run, write_timing

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def opts(ap):
    """Add rerank-specific CLI options (first-stage source run, rerank depth, output name)."""
    ap.add_argument("--source", default="Hybrid", help="first-stage run to rerank")
    ap.add_argument("--depth", type=int, default=50, help="number of top candidates to rescore")
    ap.add_argument("--name", default="Rerank", help="system name for the output run")


def main():
    """Rescore the top `depth` docs of the source run with the cross-encoder; keep the tail as-is."""
    args = cli(__doc__, opts)
    seed_all()
    hf_logging.set_verbosity_error()
    dev = device()
    model = CrossEncoder(MODEL, device=dev)
    for ds in args.datasets:
        corpus, queries, _ = load(ds)
        src = read_run(ds, args.source)
        run, times = {}, []
        for qid, docs in src.items():
            t = time.perf_counter()
            head, tail = docs[:args.depth], docs[args.depth:]
            if head:
                # Cross-encoder reads query and doc jointly and outputs one relevance logit per pair;
                # stable sort keeps first-stage order among ties.
                s = model.predict([(queries[qid], corpus[d]) for d in head], batch_size=50, show_progress_bar=False)
                head = [head[j] for j in np.argsort(-np.asarray(s), kind="stable")]
            run[qid] = head + tail
            times.append(time.perf_counter() - t)
        write_run(ds, args.name, run)
        rr_ms = 1000 * sum(times) / len(times)
        st = read_timing(ds, args.source)
        # End-to-end latency = first-stage latency + rerank latency.
        total = st["mean_ms"] + rr_ms
        write_timing(ds, args.name, {"system": args.name, "rerank_ms": rr_ms, "mean_ms": total,
                                     "setup_s": st.get("setup_s"), "source": args.source,
                                     "n_queries": len(times), "depth": args.depth, "model": MODEL,
                                     "revision": hf_revision(MODEL), "device": dev,
                                     "max_length": model.max_seq_length,
                                     "note": f"mean_ms = {args.source} end-to-end + rerank"})
        print(f"[{args.name}] {ds}: rerank {rr_ms:.2f} ms/q, end-to-end {total:.2f} ms/q", flush=True)


if __name__ == "__main__":
    main()

"""S1 BM25 baseline (fixed k1=1.2, b=0.75, zero-shot).

Source: BEIR (Thakur et al., 2021) - BM25 as the robust zero-shot baseline every
semantic method must beat on unseen data. Implementation: bm25s (Lucene variant),
English stopwords, Snowball English stemmer, top 100.
"""
import time

import bm25s
import Stemmer

from retrieval_eval.common import TOPK, cli, load, seed_all, write_run, write_timing

K1, B = 1.2, 0.75


def main():
    """Index each dataset with BM25, retrieve top-100 per query, and write the run and timing files."""
    args = cli(__doc__)
    seed_all()
    stemmer = Stemmer.Stemmer("english")
    for ds in args.datasets:
        corpus, queries, _ = load(ds)
        ids = list(corpus)  # bm25s returns integer positions; map them back to doc ids via this list
        t0 = time.perf_counter()
        tok = bm25s.tokenize([corpus[i] for i in ids], stopwords="en", stemmer=stemmer, show_progress=False)
        retriever = bm25s.BM25(k1=K1, b=B, method="lucene")
        retriever.index(tok, show_progress=False)
        index_s = time.perf_counter() - t0
        vocab = tok.vocab

        run, times, empty = {}, [], 0
        for qid, q in queries.items():
            t = time.perf_counter()
            qtok = bm25s.tokenize([q], stopwords="en", stemmer=stemmer, return_ids=False, show_progress=False)[0]
            # Tokenize as strings, then drop terms absent from the corpus vocab (bm25s cannot score
            # unknown tokens); a query left with no terms yields an empty ranking.
            qtok = [w for w in qtok if w in vocab]
            if qtok:
                res, scores = retriever.retrieve([qtok], k=min(TOPK, len(ids)), show_progress=False, n_threads=1)
                docs = [ids[d] for d, s in zip(res[0], scores[0]) if s > 0]  # drop zero-score (unmatched) docs
            else:
                docs = []
            times.append(time.perf_counter() - t)  # per-query latency (tokenize + retrieve)
            empty += not docs
            run[qid] = docs
        write_run(ds, "BM25", run)
        info = {"system": "BM25", "index_s": index_s, "mean_ms": 1000 * sum(times) / len(times),
                "n_queries": len(times), "empty_queries": empty, "k1": K1, "b": B, "method": "lucene",
                "stopwords": "bm25s 'en'", "stemmer": "Snowball english (PyStemmer)"}
        write_timing(ds, "BM25", info)
        print(f"[BM25] {ds}: index {index_s:.1f}s, {info['mean_ms']:.2f} ms/q, empty={empty}", flush=True)


if __name__ == "__main__":
    main()

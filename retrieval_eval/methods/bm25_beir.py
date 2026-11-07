"""BM25-BEIR: reproduction of BEIR's published BM25 baseline configuration.

Source: BEIR (Thakur et al., 2021), Sec. 4(i): "We use Anserini with the default Lucene
parameters (k=0.9 and b=0.4). We index the title (if available) and passage as separate
fields for documents." Reproduced here with bm25s (Lucene BM25 variant), one index per
field, field scores summed (as an Anserini multi-field query does), Porter stemmer and
English stopwords (Lucene's English analyzer uses PorterStemFilter). Top 100.
Published nDCG@10 (BEIR Table 2): SciFact 0.665, NFCorpus 0.325, FiQA 0.236, TREC-COVID 0.656.
"""
import time

import bm25s
import numpy as np
import Stemmer

from retrieval_eval.common import TOPK, cli, load, load_beir_fields, seed_all, write_run, write_timing

K1, B = 0.9, 0.4
NAME = "BM25-BEIR"


def main():
    """Build one BM25 index per field (title, body), score queries by summed field scores, write run/timing."""
    args = cli(__doc__)
    seed_all()
    stemmer = Stemmer.Stemmer("porter")
    tok = lambda texts: bm25s.tokenize(texts, stopwords="en", stemmer=stemmer, show_progress=False)
    for ds in args.datasets:
        _, queries, _ = load(ds)  # asserts Step 0 counts
        fields = load_beir_fields(ds)
        ids = list(fields)
        t0 = time.perf_counter()
        idx = []  # one BM25 retriever per non-empty field; all share the same doc order (ids)
        used = []
        for f, fname in ((0, "title"), (1, "body")):
            t = tok([fields[i][f] for i in ids])
            if not t.vocab:  # e.g. FiQA has no titles: skip the empty field
                continue
            r = bm25s.BM25(k1=K1, b=B, method="lucene")
            r.index(t, show_progress=False)
            idx.append(r)
            used.append(fname)
        index_s = time.perf_counter() - t0

        run, times, empty = {}, [], 0
        k = min(TOPK, len(ids))
        for qid, q in queries.items():
            t = time.perf_counter()
            qt = bm25s.tokenize([q], stopwords="en", stemmer=stemmer, return_ids=False, show_progress=False)[0]
            # Multi-field BM25: score every doc against each field index separately and add the
            # scores, mimicking an Anserini query over title and contents fields. Each field has its
            # own vocab, so unknown terms are filtered per field.
            s = np.zeros(len(ids), dtype=np.float32)
            for r in idx:
                qf = [w for w in qt if w in r.vocab_dict]
                if qf:
                    s += r.get_scores(qf)
            # Top-k without a full sort: argpartition picks the k best, then sort just those.
            top = np.argpartition(-s, k - 1)[:k]
            top = top[np.argsort(-s[top], kind="stable")]
            docs = [ids[j] for j in top if s[j] > 0]  # drop docs matching no query term
            times.append(time.perf_counter() - t)
            empty += not docs
            run[qid] = docs
        write_run(ds, NAME, run)
        info = {"system": NAME, "index_s": index_s, "setup_s": index_s, "mean_ms": 1000 * sum(times) / len(times),
                "n_queries": len(times), "empty_queries": empty, "k1": K1, "b": B, "method": "lucene",
                "fields": "+".join(used) + " (scores summed)", "stopwords": "bm25s 'en'", "stemmer": "Porter (PyStemmer)"}
        write_timing(ds, NAME, info)
        print(f"[{NAME}] {ds}: index {index_s:.1f}s, {info['mean_ms']:.2f} ms/q, empty={empty}", flush=True)


if __name__ == "__main__":
    main()

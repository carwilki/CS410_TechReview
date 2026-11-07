"""SPLADE: learned sparse retrieval (zero-shot).

Source: SPLADE (Formal et al., SIGIR 2021) and SPLADE v2 (Formal et al., 2021, arXiv
2109.10086). Each text is mapped to a vocabulary-sized sparse vector of term weights
(MLM logits, log(1 + ReLU), max-pooled over tokens), so queries are expanded with related
terms; score = dot product of query and document vectors.
Checkpoint: naver/splade_v2_distil (DistilSPLADE-max), the model whose BEIR results are
published in SPLADE v2, Table 2 (nDCG@10): SciFact 0.693, NFCorpus 0.334, FiQA 0.336,
TREC-COVID 0.710. The SIGIR 2021 SPLADE paper reports no BEIR results.
Max sequence length 256 for queries and documents, as in SPLADEv2's BEIR evaluation.
Exact search via a scipy CSR document matrix (no inverted-index approximation). Top 100.
"""
import time

import numpy as np
import scipy.sparse as sp
from sentence_transformers import SparseEncoder
from transformers.utils import logging as hf_logging

from retrieval_eval.common import TOPK, cli, device, hf_revision, load, seed_all, write_run, write_timing

MODEL = "naver/splade_v2_distil"
BATCH = 64
MAX_LEN = 256  # SPLADEv2 BEIR setting: "maximum sequence length 256 ... for both queries and documents" (ColBERTv2 paper, App.)


def to_csr(t):
    """Convert a torch sparse COO tensor (rows x vocab) of SPLADE term weights to a scipy CSR matrix."""
    t = t.coalesce().cpu()  # coalesce merges duplicate indices so indices()/values() are valid
    dim = t.shape[1]  # vocabulary size (30522), not the hidden size
    (r, c), v = t.indices().numpy(), t.values().numpy()
    return sp.csr_matrix((v, (r, c)), shape=(t.shape[0], dim), dtype=np.float32)


def main():
    """Encode each corpus into a sparse doc-term matrix, score queries by sparse dot product, write run/timing."""
    args = cli(__doc__)
    seed_all()
    hf_logging.set_verbosity_error()
    dev = device()
    model = SparseEncoder(MODEL, device=dev)
    model.max_seq_length = MAX_LEN
    for ds in args.datasets:
        corpus, queries, _ = load(ds)
        ids = list(corpus)
        texts = [corpus[i] for i in ids]
        t0 = time.perf_counter()
        parts = []
        for s in range(0, len(texts), 4096):  # chunk to bound memory on the large corpora
            parts.append(to_csr(model.encode_document(texts[s:s + 4096], batch_size=BATCH,
                                                      convert_to_sparse_tensor=True)))
            if s == 0:
                print(f"[SPLADE] {ds}: projected encode {(time.perf_counter() - t0) * len(texts) / min(4096, len(texts)) / 60:.1f} min",
                      flush=True)
        D = sp.vstack(parts).tocsr()  # (n_docs x vocab) document matrix, row order = ids
        encode_s = time.perf_counter() - t0

        run, times = {}, []
        k = min(TOPK, len(ids))
        for qid, q in queries.items():
            t = time.perf_counter()
            qv = to_csr(model.encode_query([q], convert_to_sparse_tensor=True))
            # Sparse mat-vec gives every doc's exact dot product with the (expanded) query vector;
            # only docs sharing a nonzero vocab dimension with the query get a positive score.
            s = (D @ qv.T).toarray().ravel()
            top = np.argpartition(-s, k - 1)[:k]
            top = top[np.argsort(-s[top], kind="stable")]
            run[qid] = [ids[j] for j in top if s[j] > 0]  # drop docs with no term overlap
            times.append(time.perf_counter() - t)
        write_run(ds, "SPLADE", run)
        info = {"system": "SPLADE", "encode_s": encode_s, "setup_s": encode_s,
                "mean_ms": 1000 * sum(times) / len(times), "n_queries": len(times), "model": MODEL,
                "revision": hf_revision(MODEL), "device": dev, "max_seq_length": model.max_seq_length,
                "doc_nnz_mean": float(D.getnnz(axis=1).mean())}
        write_timing(ds, "SPLADE", info)
        print(f"[SPLADE] {ds}: encode {encode_s:.1f}s, {info['mean_ms']:.2f} ms/q, "
              f"avg {info['doc_nnz_mean']:.0f} terms/doc", flush=True)


if __name__ == "__main__":
    main()

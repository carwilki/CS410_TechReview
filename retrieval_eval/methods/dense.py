"""S2 Dense bi-encoder retrieval (zero-shot).

Source: DPR (Karpukhin et al., 2020) - dual-encoder architecture: queries and documents
encoded separately, ranked by dot product of (normalized => cosine) embeddings. DPR
searched with FAISS; here exact numpy search is used because the corpora are small.
Zero-shot setup per BEIR (Thakur et al., 2021): no training on the test collection.
Model: BAAI/bge-small-en-v1.5, query instruction prefix per model card, no doc prefix.
"""
import time

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as hf_logging

from retrieval_eval.common import TOPK, cli, device, hf_revision, load, seed_all, write_run, write_timing

MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BATCH = 128


def main():
    """Encode each corpus with the bi-encoder, run exact cosine top-100 search per query, write run/timing."""
    args = cli(__doc__)
    seed_all()
    hf_logging.set_verbosity_error()  # silence "sequence longer than max length" warnings
    dev = device()
    model = SentenceTransformer(MODEL, device=dev)
    max_len = model.max_seq_length
    for ds in args.datasets:
        corpus, queries, _ = load(ds)
        ids = list(corpus)
        texts = [corpus[i] for i in ids]

        # Count docs longer than the model's max length (they get truncated) for the report.
        lens = [len(x) for x in model.tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]]
        n_trunc = sum(l > max_len for l in lens)
        print(f"[Dense] {ds}: {n_trunc}/{len(texts)} docs exceed {max_len} tokens", flush=True)

        # Throughput probe on a slice, then full encode (no abort, per user instruction).
        probe = texts[: min(1024, len(texts))]
        t = time.perf_counter()
        model.encode(probe, batch_size=BATCH, normalize_embeddings=True)
        rate = len(probe) / (time.perf_counter() - t)
        print(f"[Dense] {ds}: projected encode {len(texts) / rate / 60:.1f} min ({rate:.0f} docs/s)", flush=True)

        t0 = time.perf_counter()
        emb = model.encode(texts, batch_size=BATCH, normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=len(texts) > 10000).astype(np.float32)
        encode_s = time.perf_counter() - t0

        run, times = {}, []
        k = min(TOPK, len(ids))
        for qid, q in queries.items():
            t = time.perf_counter()
            qv = model.encode([QUERY_PREFIX + q], normalize_embeddings=True, convert_to_numpy=True)[0]
            s = emb @ qv  # both sides L2-normalized, so dot product = cosine similarity
            # Top-k without a full sort: argpartition picks the k best, then sort just those.
            top = np.argpartition(-s, k - 1)[:k]
            top = top[np.argsort(-s[top], kind="stable")]
            times.append(time.perf_counter() - t)
            run[qid] = [ids[j] for j in top]
        write_run(ds, "Dense", run)
        info = {"system": "Dense", "encode_s": encode_s, "mean_ms": 1000 * sum(times) / len(times),
                "n_queries": len(times), "model": MODEL, "revision": hf_revision(MODEL), "device": dev,
                "max_seq_length": max_len, "docs_truncated": n_trunc, "n_docs": len(texts),
                "doc_tokens_mean": float(np.mean(lens)), "doc_tokens_median": float(np.median(lens))}
        write_timing(ds, "Dense", info)
        print(f"[Dense] {ds}: encode {encode_s:.1f}s, {info['mean_ms']:.2f} ms/q", flush=True)


if __name__ == "__main__":
    main()

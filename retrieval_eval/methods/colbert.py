"""ColBERTv2 late-interaction retrieval with a PLAID index (zero-shot).

Sources: ColBERT (Khattab & Zaharia, SIGIR 2020): per-token query and document embeddings,
score = sum over query tokens of the max similarity with any document token (MaxSim).
ColBERTv2 (Santhanam et al., NAACL 2022): residual compression; BEIR setup per its appendix:
"set the maximum document length of ... ColBERTv2 to 300 tokens on BEIR", "default query
maximum sequence length ... 32 tokens", embeddings compressed "to b = 2 bits per dimension".
PLAID (Santhanam et al., 2022): centroid-interaction / pruning search engine over that index.
Published ColBERTv2 nDCG@10 (ColBERTv2 Table 5a): SciFact 69.3, NFCorpus 33.8, FiQA 35.6,
TREC-COVID 73.8. Implementation: PyLate (colbert-ir/colbertv2.0, PLAID index with nbits=2).
Runs in its own venv (.venv-colbert) because PyLate pins older torch/transformers.
"""
import shutil
import time

from pylate import indexes, models, retrieve

from retrieval_eval.common import HERE, TOPK, cli, device, hf_revision, load, seed_all, write_run, write_timing

MODEL = "colbert-ir/colbertv2.0"
DOC_LEN, QUERY_LEN, NBITS = 300, 32, 2
BATCH = 64


def main():
    """Encode docs into per-token embeddings, build a PLAID index, retrieve top-100 per query, write run/timing."""
    args = cli(__doc__)
    seed_all()
    dev = device()
    # Truncation lengths follow the ColBERTv2 BEIR setup (300 doc / 32 query tokens).
    model = models.ColBERT(model_name_or_path=MODEL, device=dev, document_length=DOC_LEN, query_length=QUERY_LEN)
    for ds in args.datasets:
        corpus, queries, _ = load(ds)
        ids = list(corpus)
        folder = HERE / "indexes" / "colbert"
        shutil.rmtree(folder / ds, ignore_errors=True)  # rebuild from scratch so timing and size are fresh
        t0 = time.perf_counter()
        # One embedding matrix per document (one row per token), not a single vector.
        demb = model.encode([corpus[i] for i in ids], batch_size=BATCH, is_query=False,
                            show_progress_bar=len(ids) > 10000)
        encode_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        # PLAID clusters token embeddings into centroids and stores each token as a centroid id
        # plus a residual compressed to NBITS bits/dim; search prunes candidates via centroids
        # before exact MaxSim scoring.
        index = indexes.PLAID(index_folder=str(folder), index_name=ds, override=True, nbits=NBITS,
                              show_progress=False)
        index.add_documents(documents_ids=ids, documents_embeddings=demb)
        index_s = time.perf_counter() - t1
        del demb  # raw embeddings are now in the compressed index; free the memory
        retriever = retrieve.ColBERT(index=index)

        run, times = {}, []
        for qid, q in queries.items():
            t = time.perf_counter()
            qemb = model.encode([q], is_query=True, show_progress_bar=False)
            res = retriever.retrieve(queries_embeddings=qemb, k=TOPK)[0]  # list of {"id", "score"} for the one query
            run[qid] = [r["id"] for r in res]
            times.append(time.perf_counter() - t)
        write_run(ds, "ColBERTv2", run)
        # On-disk index size, to compare storage cost against single-vector and sparse methods.
        size_gb = sum(f.stat().st_size for f in (folder / ds).rglob("*") if f.is_file()) / 1e9
        info = {"system": "ColBERTv2", "encode_s": encode_s, "index_s": index_s, "setup_s": encode_s + index_s,
                "mean_ms": 1000 * sum(times) / len(times), "n_queries": len(times), "model": MODEL,
                "revision": hf_revision(MODEL), "device": dev, "document_length": DOC_LEN,
                "query_length": QUERY_LEN, "nbits": NBITS, "index": "PyLate PLAID", "index_gb": size_gb}
        write_timing(ds, "ColBERTv2", info)
        print(f"[ColBERTv2] {ds}: encode {encode_s:.1f}s, index {index_s:.1f}s ({size_gb:.2f} GB), "
              f"{info['mean_ms']:.2f} ms/q", flush=True)


if __name__ == "__main__":
    main()

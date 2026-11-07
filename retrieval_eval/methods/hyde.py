"""HyDE: Hypothetical Document Embeddings (Gao et al., ACL 2023) - zero-shot dense retrieval.

An instruction-following LLM writes N hypothetical documents for the query; the query vector
is the mean of the encoder embeddings of the query and the N documents (paper Eq. 8; official
code texttron/hyde hyde.py encode() averages [query] + hypothesis_documents). Settings from
the official generator: n=8, temperature=0.7, max_tokens=512, stop '\\n\\n\\n'. Prompts are the
paper's per-task instructions (App. A.1; promptor.py). NFCorpus is not in the paper: it uses
the TREC-COVID (scientific) prompt - our choice.

Systems written:
  Contriever       facebook/contriever (unsupervised, mean pooling) alone - the paper's baseline
  HyDE-Contriever  HyDE with Contriever - reproduction of the paper's HyDE row
  HyDE-bge         HyDE with bge-small-en-v1.5 - our system, compared with Dense
  HyDE-Contriever-100w  our deviation: markdown stripped and each hypothetical doc cut to 100
                   words before encoding. gpt-oss writes ~500-word markdown answers; Contriever's
                   unnormalised mean-pooled vectors are swamped by them (FiQA 0.063 faithful).
LLM: gpt-oss-120b via Databricks Unity Gateway (paper used text-davinci-003).
Published nDCG@10 (HyDE Table 2): Contriever SciFact 64.9 / FiQA 24.5 / TREC-COVID 27.3;
HyDE 69.1 / 27.3 / 59.3. NFCorpus not reported.
"""
import json
import re
import time

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as hf_logging

from retrieval_eval.common import HERE, TOPK, cli, device, hf_revision, load, seed_all, write_run, write_timing
from retrieval_eval.llm import GPT_OSS, LLM

N, TEMP, MAX_TOKENS, STOP = 8, 0.7, 512, "\n\n\n"
PROMPTS = {  # verbatim from texttron/hyde src/hyde/promptor.py
    "scifact": "Please write a scientific paper passage to support/refute the claim.\nClaim: {}\nPassage:",
    "trec-covid": "Please write a scientific paper passage to answer the question.\nQuestion: {}\nPassage:",
    "fiqa": "Please write a financial article passage to answer the question.\nQuestion: {}\nPassage:",
    "nfcorpus": "Please write a scientific paper passage to answer the question.\nQuestion: {}\nPassage:",
}
# encoder key -> (HF model id, query prefix). Only bge uses an instruction prefix and normalization.
ENCODERS = {
    "bge": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: "),
    "contriever": ("facebook/contriever", ""),
}
EMB = HERE / "emb_cache"  # corpus embeddings cached across runs (encoding is the slow part)
SHORT_WORDS = 100


def shorten(t):
    """Strip markdown (table rows, emphasis/heading/code marks) and keep the first SHORT_WORDS words.

    Used only for the HyDE-Contriever-100w deviation, so long LLM answers don't dominate the mean vector.
    """
    t = re.sub(r"^\s*\|.*\|\s*$", " ", t, flags=re.M)  # markdown table rows
    t = re.sub(r"[*#_`>]+", " ", t)
    return " ".join(t.split()[:SHORT_WORDS])


def opts(ap):
    """Add the --generate-only CLI flag."""
    ap.add_argument("--generate-only", action="store_true", help="only generate (and cache) hypothetical docs")


def generate(llm, ds, queries):
    """Generate N hypothetical documents per query with the LLM.

    Returns {qid: [doc_text, ...]} (N entries, possibly empty strings on failure). Each sample
    gets a distinct `_salt` so the LLM cache stores N separate samples instead of one.
    """
    jobs = [(PROMPTS[ds].format(q), f"hyde:{ds}:{qid}:{i}",
             {"temperature": TEMP, "max_output_tokens": MAX_TOKENS + 256,  # + small budget for gpt-oss reasoning
              "reasoning": {"effort": "low"}, "_salt": i})
            for qid, q in queries.items() for i in range(N)]
    outs = llm.map(jobs, desc=f"HyDE {ds}")
    gen = {}
    for (_, tag, _), text in zip(jobs, outs):
        qid = tag.split(":")[2]  # tag = "hyde:{ds}:{qid}:{i}"
        # Apply the official stop sequence client-side (truncate at the first triple newline).
        gen.setdefault(qid, []).append(text.split(STOP)[0].strip())
    return gen


def corpus_embeddings(model, name, ds, ids, texts):
    """Return (doc_embeddings, encode_seconds) for a corpus, loading from emb_cache when the row count matches.

    bge embeddings are L2-normalized; Contriever's are left raw, as in the paper's dot-product setup.
    """
    EMB.mkdir(exist_ok=True)
    f = EMB / f"{ds}.{name}.npy"
    if f.exists():
        e = np.load(f)
        if e.shape[0] == len(ids):  # cached: report the encode time measured when it was built
            t = EMB / f"{ds}.{name}.json"
            return e, json.loads(t.read_text())["encode_s"] if t.exists() else float("nan")
    t = time.perf_counter()
    e = model.encode(texts, batch_size=128, normalize_embeddings=(name == "bge"), convert_to_numpy=True,
                     show_progress_bar=len(texts) > 10000).astype(np.float32)
    np.save(f, e)
    s = time.perf_counter() - t
    (EMB / f"{ds}.{name}.json").write_text(json.dumps({"encode_s": s}))
    return e, s


def search(emb, qv, ids, k):
    """Exact inner-product search: return the ids of the k docs with the highest emb @ qv."""
    s = emb @ qv
    top = np.argpartition(-s, k - 1)[:k]
    return [ids[j] for j in top[np.argsort(-s[top], kind="stable")]]


def main():
    """Generate (or load cached) hypothetical docs, then for each encoder write plain, HyDE and 100w runs."""
    args = cli(__doc__, opts)
    seed_all()
    hf_logging.set_verbosity_error()
    llm = LLM(GPT_OSS)
    # Phase 1: all LLM generation up front (cached on disk), before any model is loaded.
    gens = {}
    for ds in args.datasets:
        _, queries, _ = load(ds)
        gens[ds] = generate(llm, ds, queries)
        empty = sum(1 for v in gens[ds].values() for t in v if not t)
        print(f"[HyDE] {ds}: generated {sum(len(v) for v in gens[ds].values())} docs ({empty} empty)", flush=True)
    stats = llm.report("HyDE generation")
    if args.generate_only:
        return

    # Phase 2: per encoder, embed corpus once and build all its query variants from the same doc vectors.
    dev = device()
    for enc, (mid, qprefix) in ENCODERS.items():
        model = SentenceTransformer(mid, device=dev)
        norm = enc == "bge"
        for ds in args.datasets:
            corpus, queries, _ = load(ds)
            ids = list(corpus)
            emb, enc_s = corpus_embeddings(model, enc, ds, ids, [corpus[i] for i in ids])
            k = min(TOPK, len(ids))
            # LLM latency per query: the N samples are independent, so time = slowest sample
            rec = {r["tag"]: r["seconds"] for r in llm.cache.values() if r.get("tag", "").startswith(f"hyde:{ds}:")}
            runs = {"hyde": {}, "plain": {}, "short": {}}
            t_hyde, t_plain, t_short = [], [], []
            for qid, q in queries.items():
                t = time.perf_counter()
                qe = model.encode([qprefix + q], normalize_embeddings=norm, convert_to_numpy=True)[0]
                runs["plain"][qid] = search(emb, qe, ids, k)
                t_plain.append(time.perf_counter() - t)
                t = time.perf_counter()
                docs = [d for d in gens[ds][qid] if d]  # skip empty generations
                # With no usable docs, an empty (0, dim) array makes the mean below fall back to the query alone.
                de = model.encode(docs, normalize_embeddings=norm, convert_to_numpy=True) if docs else np.zeros((0, len(qe)))
                qv = np.vstack([qe[None, :], de]).mean(axis=0)  # Eq. 8: mean of f(q) and f(d_hat_k)
                runs["hyde"][qid] = search(emb, qv, ids, k)
                llm_s = max((rec.get(f"hyde:{ds}:{qid}:{i}", 0.0) for i in range(N)), default=0.0)
                t_hyde.append(time.perf_counter() - t + llm_s)
                if enc == "contriever":
                    # Deviation: same averaging, but each hypothetical doc is shortened first.
                    t = time.perf_counter()
                    de = model.encode([shorten(d) for d in docs], convert_to_numpy=True) if docs else np.zeros((0, len(qe)))
                    runs["short"][qid] = search(emb, np.vstack([qe[None, :], de]).mean(axis=0), ids, k)
                    t_short.append(time.perf_counter() - t + llm_s)
            # Map each run variant to its output system name (None = don't write).
            names = {"hyde": "HyDE-bge" if enc == "bge" else "HyDE-Contriever", "plain": None if enc == "bge" else "Contriever",
                     "short": None if enc == "bge" else "HyDE-Contriever-100w"}
            for kind, name in names.items():
                if name is None:
                    continue  # plain bge = Dense, already produced by retrieval_eval.methods.dense
                times = {"hyde": t_hyde, "plain": t_plain, "short": t_short}[kind]
                write_run(ds, name, runs[kind])
                write_timing(ds, name, {"system": name, "mean_ms": 1000 * float(np.mean(times)), "encode_s": enc_s,
                                        "setup_s": enc_s, "n_queries": len(times), "encoder": mid,
                                        "encoder_revision": hf_revision(mid), "device": dev,
                                        **({"llm": GPT_OSS, "n_samples": N, "temperature": TEMP,
                                            "note": "mean_ms includes the slowest of the N parallel LLM calls",
                                            "llm_stats": stats} if kind != "plain" else {})})
                print(f"[{name}] {ds}: {1000 * np.mean(times):.1f} ms/q (corpus encode {enc_s:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

"""Geometry of HyDE query vectors: where do the query, the hypothetical passages and their mean land
relative to the relevant documents?

Uses only cached artefacts (no LLM calls): the gpt-oss generations in llm_cache/ (tags
hyde:{ds}:{qid}:{i}) and the corpus embeddings in emb_cache/. For each dataset and encoder it reports
mean cosine similarity to the judged-relevant documents and the rank of the first relevant document,
for (a) the query vector f(q), (b) each hypothetical passage f(d_hat), (c) the HyDE vector
mean(f(q), f(d_hat_1..8)). For Contriever it also reports vector norms, which explain the failed
reproduction. Writes hyde_geometry.json, report/hyde_geometry.tex (table), report/hyde_scatter.pdf
and report/hyde_pca.pdf.

    .venv/bin/python -m retrieval_eval.evaluation.hyde_geometry
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np
from sentence_transformers import SentenceTransformer

from retrieval_eval.common import BEIR_SETS, HERE, device, load, seed_all
import argparse

from retrieval_eval.methods.hyde import EMB, ENCODERS, GENERATORS, STOP, shorten

REPORT = HERE / "report"
# Validated default categorical slots (dataviz palette): blue, orange, aqua; gray for background docs.
C_Q, C_H, C_R, C_BG, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#c9c8c2", "#0b0b0b", "#52514e"
EXAMPLE = ("scifact", None)  # dataset for the PCA picture; qid chosen automatically (largest rank gain)


SFX = ""  # output-file suffix: "" for gpt-oss (the report's figures), "-llama" for the Llama generator


def generations(ds, g=GENERATORS["gpt-oss"]):
    """Read the cached HyDE passages for `ds` straight from the JSONL cache: {qid: [text]*N}."""
    gen = {}
    with open(HERE / "llm_cache" / f"{g['model']}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            t = r.get("tag", "")
            if t.startswith(f"{g['tag']}:{ds}:"):
                _, _, qid, i = t.split(":")
                gen.setdefault(qid, {})[int(i)] = r["text"].split(STOP)[0].strip()  # last record wins, as in the run
    return {q: [v[i] for i in sorted(v) if v[i]] for q, v in gen.items()}


def unit(x):
    """L2-normalize rows (or a single vector)."""
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def first_rel_rank(emb, v, rel_idx):
    """1-based rank of the best-ranked relevant doc when the corpus is scored by emb @ v."""
    s = emb @ v
    return int((s > s[rel_idx].max()).sum()) + 1


def analyse(enc, model, qprefix, ds, gens):
    """Per-query statistics for one encoder and dataset; also returns vectors for the plots."""
    corpus, queries, qrels = load(ds)
    ids = list(corpus)
    pos = {d: j for j, d in enumerate(ids)}
    emb = np.load(EMB / f"{ds}.{enc}.npy")
    norm = enc == "bge"
    embu = emb if norm else unit(emb)  # cosine needs unit doc vectors; ranking uses raw emb as in the run
    rng = np.random.default_rng(0)
    bg = embu[rng.choice(len(ids), size=min(2000, len(ids)), replace=False)]
    rows, keep = [], {}
    qids = [q for q in queries if q in qrels and any(g > 0 for g in qrels[q].values()) and gens.get(q)]
    qe = model.encode([qprefix + queries[q] for q in qids], normalize_embeddings=norm, convert_to_numpy=True)
    for q, fq in zip(qids, qe):
        hyps = gens[q]
        fh = model.encode(hyps, normalize_embeddings=norm, convert_to_numpy=True)
        v = np.vstack([fq[None], fh]).mean(0)  # HyDE vector, Eq. 8
        rel = [pos[d] for d, g in qrels[q].items() if g > 0 and d in pos]
        R = embu[rel]
        r = {"qid": q,
             "cos_q_rel": float((R @ unit(fq)).mean()),
             "cos_h_rel": float((R @ unit(fh).T).mean()),
             "cos_v_rel": float((R @ unit(v)).mean()),
             "cos_q_bg": float((bg @ unit(fq)).mean()),
             "cos_v_bg": float((bg @ unit(v)).mean()),
             "cos_q_h": float((unit(fh) @ unit(fq)).mean()),
             "cos_h_h": float(np.mean((unit(fh) @ unit(fh).T)[np.triu_indices(len(fh), 1)])) if len(fh) > 1 else float("nan"),
             "rank_q": first_rel_rank(emb, fq, rel),
             "rank_v": first_rel_rank(emb, v, rel),
             "words_h": float(np.mean([len(h.split()) for h in hyps]))}
        if not norm:
            # Norms explain why long passages dominate an unnormalized mean, and what shortening does.
            fs = model.encode([shorten(h) for h in hyps], convert_to_numpy=True)
            r |= {"norm_q": float(np.linalg.norm(fq)), "norm_h": float(np.linalg.norm(fh, axis=1).mean()),
                  "norm_h100": float(np.linalg.norm(fs, axis=1).mean()),
                  "norm_doc": float(np.linalg.norm(emb[rel], axis=1).mean()),
                  "cos_v_q": float(unit(v) @ unit(fq))}
        rows.append(r)
        if norm and ds == EXAMPLE[0]:
            keep[q] = (fq, fh, v, rel, queries[q], hyps)
    return rows, keep, embu


def summarise(rows):
    """Dataset-level means/medians of the per-query statistics."""
    a = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0] if k != "qid"}
    s = {k: float(np.nanmean(v)) for k, v in a.items() if not k.startswith("rank")}
    s |= {"median_rank_q": float(np.median(a["rank_q"])), "median_rank_v": float(np.median(a["rank_v"])),
          "frac_v_closer": float(np.mean(a["cos_v_rel"] > a["cos_q_rel"])),
          "frac_rank_better": float(np.mean(a["rank_v"] < a["rank_q"])),
          "frac_rank_worse": float(np.mean(a["rank_v"] > a["rank_q"])), "n": len(rows)}
    return s


def scatter(per_ds):
    """One panel per dataset: rank of the first relevant doc with f(q) vs with the HyDE vector (bge, log axes).

    Ranks, not raw cosines: the HyDE vector is closer to *every* document (anisotropy), so only the
    relative order of relevant vs other documents matters for search.
    """
    fig, axes = plt.subplots(1, len(per_ds), figsize=(10, 2.9), constrained_layout=True)
    for ax, (ds, rows) in zip(axes, per_ds.items()):
        x = np.array([r["rank_q"] for r in rows]); y = np.array([r["rank_v"] for r in rows])
        jit = np.random.default_rng(0).uniform(0.9, 1.1, (2, len(x)))  # separate the many ties at rank 1
        hi = max(x.max(), y.max(), 20) * 1.5
        ax.plot([0.8, hi], [0.8, hi], color=INK2, lw=1, ls="--")
        ax.scatter(x * jit[0], y * jit[1], s=10, color=C_Q, alpha=0.5, linewidths=0)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.8, hi); ax.set_ylim(0.8, hi); ax.set_aspect("equal")
        for a in (ax.xaxis, ax.yaxis):  # plain 1, 10, 100 labels; no minor labels
            a.set_major_locator(LogLocator(base=10)); a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            a.set_minor_formatter(NullFormatter())
        ax.set_title(f"{ds}: {np.mean(y < x):.0%} better, {np.mean(y > x):.0%} worse", fontsize=9, color=INK)
        ax.set_xlabel("rank with query vector", fontsize=7, color=INK2)
        ax.tick_params(labelsize=6, colors=INK2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("rank with HyDE vector", fontsize=7, color=INK2)
    fig.savefig(REPORT / f"hyde_scatter{SFX}.pdf")


def pca_plot(keep, rows, embu):
    """2-D PCA picture of one SciFact query: background docs, relevant docs, query, passages, HyDE vector."""
    by = {r["qid"]: r for r in rows}
    # Illustrative example: the query whose first relevant doc moves up the most (stated in the caption).
    q = max(keep, key=lambda k: np.log(by[k]["rank_q"]) - np.log(by[k]["rank_v"]))
    fq, fh, v, rel, qtext, hyps = keep[q]
    rng = np.random.default_rng(0)
    top_q = np.argsort(-(embu @ unit(fq)))[:10]  # what plain dense search returns
    top_v = np.argsort(-(embu @ unit(v)))[:10]   # what HyDE search returns
    near = np.argsort(-(embu @ unit(v)))[:300]  # local neighbourhood so the plane shows the relevant region
    bg = embu[np.unique(np.r_[near, top_q, rng.choice(len(embu), 300, replace=False)])]
    pts = np.vstack([bg, embu[rel], unit(fq)[None], unit(fh), unit(v)[None]])
    mu = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - mu, full_matrices=False)
    P = lambda x: (np.atleast_2d(x) - mu) @ vt[:2].T
    fig, ax = plt.subplots(figsize=(6.6, 3.9), constrained_layout=True)
    ax.scatter(*P(bg).T, s=6, color=C_BG, linewidths=0, label="other documents")
    ax.scatter(*P(embu[top_q]).T, s=70, facecolors="none", edgecolors=C_Q, lw=1.1, label="top 10 for $f(q)$")
    ax.scatter(*P(embu[top_v]).T, s=120, facecolors="none", edgecolors=INK, lw=1.1, label="top 10 for HyDE vector")
    ax.scatter(*P(embu[rel]).T, s=60, color=C_R, marker="s", edgecolors="white", lw=1.5, label="relevant document")
    ax.scatter(*P(unit(fh)).T, s=34, color=C_H, edgecolors="white", lw=1.2, label="hypothetical passages $f(\\hat d_k)$")
    pq, pv = P(unit(fq))[0], P(unit(v))[0]
    ax.scatter(*pq, s=90, color=C_Q, marker="o", edgecolors="white", lw=1.5, label="query $f(q)$")
    ax.scatter(*pv, s=200, color=INK, marker="*", edgecolors="white", lw=1, label="HyDE vector (mean)")
    for h in P(unit(fh)):
        ax.annotate("", xy=pv, xytext=h, arrowprops=dict(arrowstyle="-", color=C_H, lw=0.6, alpha=0.6))
    ax.annotate("", xy=pv, xytext=pq, arrowprops=dict(arrowstyle="-|>", color=C_Q, lw=1.4))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(C_BG)
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_title(f"SciFact query {q}: first relevant doc rank {by[q]['rank_q']} $\\to$ {by[q]['rank_v']}",
                 fontsize=9, color=INK)
    fig.savefig(REPORT / f"hyde_pca{SFX}.pdf")
    return {"top_q": [int(j in rel) for j in top_q], "top_v": [int(j in rel) for j in top_v], "qid": q, "query": qtext, "passage": hyps[0], "rank_q": by[q]["rank_q"], "rank_v": by[q]["rank_v"],
            "cos_q_rel": by[q]["cos_q_rel"], "cos_v_rel": by[q]["cos_v_rel"]}


def table(summ):
    """LaTeX table of the geometry statistics (bge and Contriever)."""
    L = [r"\begin{table}[t]\centering\small", r"\resizebox{\linewidth}{!}{%",
         r"\begin{tabular}{llrrrrrrrr}\toprule",
         r" & & \multicolumn{3}{c}{mean cos to relevant docs} & & \multicolumn{2}{c}{median rank of 1st rel.} & \% queries & passage\\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){7-8}",
         r"Encoder & Dataset & $f(q)$ & $f(\hat d)$ & HyDE & cos($f(q),f(\hat d)$) & $f(q)$ & HyDE & rank better & words\\\midrule"]
    for enc in summ:
        for ds, s in summ[enc].items():
            L.append(f"{enc} & {ds} & {s['cos_q_rel']:.3f} & {s['cos_h_rel']:.3f} & {s['cos_v_rel']:.3f} & "
                     f"{s['cos_q_h']:.3f} & {s['median_rank_q']:.0f} & {s['median_rank_v']:.0f} & "
                     f"{100 * s['frac_rank_better']:.0f} & {s['words_h']:.0f}\\\\")
        L.append(r"\midrule" if enc != list(summ)[-1] else r"\bottomrule")
    L += [r"\end{tabular}}",
          r"\caption{HyDE geometry, computed from the cached generations. Cosines use unit vectors; ranks use the "
          r"scoring of the actual run. ``\% rank better'' is the share of queries whose first relevant document "
          r"moves up when $f(q)$ is replaced by the HyDE vector.}\label{tab:hydegeo" + SFX + "}", r"\end{table}"]
    (REPORT / f"hyde_geometry{SFX}.tex").write_text("\n".join(L) + "\n")


def main():
    """Analyse both encoders on all four datasets, then write the figures, table and JSON summary."""
    seed_all()
    dev = device()
    summ, out, per_ds_bge, example = {}, {}, {}, None
    global SFX
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llm", choices=list(GENERATORS), default="gpt-oss", help="which generator's cached passages")
    g = GENERATORS[ap.parse_args().llm]
    SFX = g["suffix"]
    gens = {ds: generations(ds, g) for ds in BEIR_SETS}
    for enc, (mid, qprefix) in ENCODERS.items():
        model = SentenceTransformer(mid, device=dev)
        summ[enc], out[enc] = {}, {}
        for ds in BEIR_SETS:
            rows, keep, embu = analyse(enc, model, qprefix, ds, gens[ds])
            summ[enc][ds], out[enc][ds] = summarise(rows), rows
            print(enc, ds, json.dumps({k: round(v, 3) for k, v in summ[enc][ds].items()}), flush=True)
            if enc == "bge":
                per_ds_bge[ds] = rows
                if keep:
                    example = pca_plot(keep, rows, embu)
    scatter(per_ds_bge)
    table(summ)
    (HERE / f"hyde_geometry{SFX}.json").write_text(json.dumps({"summary": summ, "example": example}, indent=1))
    print("example:", json.dumps(example)[:600])


if __name__ == "__main__":
    main()

"""Evaluate all runs against the BM25 baseline and against published results.

Protocol per BEIR (Thakur et al., 2021): nDCG@10 primary, Recall@100 secondary (plus BEIR's
capped Recall@100 for comparison with its Table 9), P@10, AP@100. Paired t-test (scipy
ttest_rel) on per-query nDCG@10 vs BM25 - our addition; BEIR reports no significance tests
(Smucker et al., 2007 support the t-test for IR metrics). If qrels_aug/{ds}.tsv exists (LLM
labels for judgment holes, llm_judge.py), every system is also scored on the augmented qrels.

Outputs (project root): results.csv, per_query.csv, paired_tests.csv, results_augmented.csv,
published_comparison.csv, eval_info.json, examples.md, RESULT.md, ndcg10.png, and report/tables.tex
+ report/ndcg10.pdf.
"""
import importlib.metadata as md
import json
import platform
import subprocess

import ir_measures
import matplotlib
import numpy as np
import pandas as pd
from ir_measures import AP, P, R, nDCG
from scipy.stats import ttest_rel

# Headless backend: figures are only saved to disk, never shown.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from retrieval_eval.common import HERE, RUNS, cli, load, qid_key, read_run, read_timing  # noqa: E402
from retrieval_eval.evaluation.published import NDCG10, RECALL100  # noqa: E402

# Display order of systems in every table/plot; only systems with runs for all datasets are kept.
ORDER = ["BM25", "BM25-BEIR", "SPLADE", "Dense", "Contriever", "ColBERTv2", "HyDE-bge", "HyDE-Contriever", "HyDE-Contriever-100w",
         "Hybrid", "Rerank", "BM25+CE", "RankGPT"]
# Method family per system, used for the table column and bar colors.
FAMILY = {"BM25": "Lexical", "BM25-BEIR": "Lexical", "SPLADE": "Learned sparse", "Dense": "Dense",
          "Contriever": "Dense", "ColBERTv2": "Late interaction", "HyDE-bge": "LLM", "HyDE-Contriever": "LLM", "HyDE-Contriever-100w": "LLM",
          "Hybrid": "Hybrid", "Rerank": "Reranker", "BM25+CE": "Reranker", "RankGPT": "LLM"}
FAM_COLOR = {"Lexical": "#2a78d6", "Learned sparse": "#eb6834", "Dense": "#1baf7a", "Late interaction": "#eda100",
             "Hybrid": "#e87ba4", "Reranker": "#6250d6", "LLM": "#e34948"}
# Per-query measures computed by ir_measures (relevance = grade >= 1).
MEASURES = {"nDCG@10": nDCG @ 10, "P@10": P @ 10, "R@100": R @ 100, "AP@100": AP @ 100}
# TIE: |delta nDCG@10| below this counts as a tie in win/tie/loss; ALPHA: significance level.
TIE, ALPHA = 0.01, 0.05


def systems_for(dss):
    """Return systems (in ORDER) that have a run file runs/{ds}.{system}.trec for every dataset in dss."""
    # Strip the "{ds}." prefix and ".trec" suffix to recover the system name.
    have = [{p.name[len(ds) + 1:-len(".trec")] for p in RUNS.glob(f"{ds}.*.trec")} for ds in dss]
    common_ = set.intersection(*have)
    return [s for s in ORDER if s in common_]


def load_aug(ds):
    """Load augmented qrels qrels_aug/{ds}.tsv (human + LLM hole labels) as {qid: {docid: grade}}.

    Returns None when llm_judge.py --holes has not been run for this dataset.
    """
    f = HERE / "qrels_aug" / f"{ds}.tsv"
    if not f.exists():
        return None
    q = {}
    with open(f) as fh:
        # Skip the TSV header row.
        next(fh)
        for line in fh:
            a, b, c = line.rstrip("\n").split("\t")
            q.setdefault(a, {})[b] = int(c)
    return q


def evaluate(qrels, runs, evq):
    """Score every run on the evaluation queries evq against qrels.

    Args:
        qrels: {qid: {docid: grade}}.
        runs: {system: {qid: [docid, ...]}} in rank order.
        evq: query ids to evaluate (those with at least one relevant doc).
    Returns:
        {system: {metric: {qid: value}}} for MEASURES plus J@10 and Rcap@100.
    """
    ql = [ir_measures.Qrel(q, d, g) for q in evq for d, g in qrels[q].items()]
    inv = {v: k for k, v in MEASURES.items()}
    per = {}
    for s, run in runs.items():
        # Runs carry only rank order; synthesize descending scores so ir_measures ranks docs exactly as listed.
        sd = [ir_measures.ScoredDoc(q, d, float(len(run[q]) - r)) for q in evq if q in run for r, d in enumerate(run[q])]
        vals = {m: {q: 0.0 for q in evq} for m in MEASURES}  # missing run => 0
        # Queries a system returned nothing for keep 0.0 instead of being dropped from the mean.
        for m in ir_measures.iter_calc(list(MEASURES.values()), ql, sd):
            vals[inv[m.measure]][m.query_id] = m.value
        # J@10 (judged@10): share of the top 10 with any judgment (any grade, incl. 0). A diagnostic of
        # judgment holes: low J@10 means the metric may undercount relevant but unjudged docs.
        vals["J@10"] = {q: sum(d in qrels[q] for d in run.get(q, [])[:10]) / 10 for q in evq}
        # Capped Recall@100 (BEIR Table 9): divide by min(|rel|, 100) so queries with >100 relevant docs
        # can still reach 1.0. evq guarantees |rel| > 0, so no division by zero.
        rel = {q: {d for d, g in qrels[q].items() if g > 0} for q in evq}
        vals["Rcap@100"] = {q: len(rel[q] & set(run.get(q, [])[:100])) / min(len(rel[q]), 100) for q in evq}
        per[s] = vals
    return per


def paired(per, evq, systems, ds, tag):
    """Paired two-sided t-test of each system's per-query nDCG@10 against BM25.

    Returns rows (tag, ds, system, mean_delta, p_value, wins, ties, losses), where win/loss means
    delta >= TIE / <= -TIE and tie means |delta| < TIE. tag names the qrels used (original/augmented).
    """
    base = np.array([per["BM25"]["nDCG@10"][q] for q in evq])
    out = []
    for s in systems:
        if s == "BM25":
            continue
        x = np.array([per[s]["nDCG@10"][q] for q in evq])
        d = x - base
        # ttest_rel is undefined (NaN) when every difference is zero; treat identical runs as p = 1.
        p = ttest_rel(x, base).pvalue if np.any(d) else 1.0
        out.append((tag, ds, s, d.mean(), p, int((d >= TIE).sum()), int((np.abs(d) < TIE).sum()), int((d <= -TIE).sum())))
    return out


def fmt_p(p):
    """Format a p-value for display, collapsing very small values to '<0.001'."""
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def main():
    """Evaluate all systems on all datasets and write every CSV/JSON/Markdown/LaTeX/figure output."""
    args = cli(__doc__)
    dss = args.datasets
    systems = systems_for(dss)
    rows, pq_rows, stats, info, aug_rows, examples = [], [], [], {}, [], []
    for ds in dss:
        corpus, queries, qrels = load(ds)
        runs = {s: read_run(ds, s) for s in systems}
        # Evaluate only queries with at least one relevant (grade > 0) judgment, as BEIR does.
        evq = sorted((q for q, v in qrels.items() if any(g > 0 for g in v.values())), key=qid_key)
        per = evaluate(qrels, runs, evq)
        timing = {s: read_timing(ds, s) for s in systems}
        # Per-dataset metadata, saved to eval_info.json and used in RESULT.md.
        info[ds] = {"n_docs": len(corpus), "n_queries": len(queries), "n_eval": len(evq),
                    "has_grade0": any(g <= 0 for v in qrels.values() for g in v.values()),
                    "avg_rel_per_query": float(np.mean([sum(g > 0 for g in qrels[q].values()) for q in evq])),
                    "timing": timing}
        # Long-format result rows: one (dataset, system, metric, mean value) per metric.
        for s in systems:
            for m in [*MEASURES, "Rcap@100", "J@10"]:
                rows.append((ds, s, m, float(np.mean(list(per[s][m].values())))))
            pq_rows += [(ds, q, s, per[s]["nDCG@10"][q]) for q in evq]
            t = timing[s]
            setup = t.get("setup_s")
            if setup is None:  # older timing files
                setup = (timing["BM25"]["index_s"] + timing["Dense"]["encode_s"] if s in ("Hybrid", "Rerank")
                         else t.get("index_s", t.get("encode_s")))
            rows += [(ds, s, "latency_ms", t["mean_ms"]), (ds, s, "setup_s", setup)]
        # Significance vs BM25 on the original human qrels.
        stats += paired(per, evq, systems, ds, "original")
        # Rescore the same runs on augmented qrels (LLM labels filling top-10 holes) when available.
        aug = load_aug(ds)
        if aug is not None:
            pa = evaluate(aug, runs, evq)
            for s in systems:
                aug_rows.append((ds, s, float(np.mean(list(pa[s]["nDCG@10"].values()))),
                                 float(np.mean(list(pa[s]["J@10"].values())))))
            stats += paired(pa, evq, systems, ds, "augmented")
        # Per-query Dense - BM25 delta; keep the 3 largest wins and 3 largest losses for examples.md.
        dd = {q: per["Dense"]["nDCG@10"][q] - per["BM25"]["nDCG@10"][q] for q in evq}
        order = sorted(evq, key=lambda q: dd[q])
        examples.append((ds, queries, qrels, corpus, runs, per, order[::-1][:3], order[:3], dd))

    res = pd.DataFrame(rows, columns=["dataset", "system", "metric", "value"])
    res.to_csv(HERE / "results.csv", index=False)
    pd.DataFrame(pq_rows, columns=["dataset", "qid", "system", "ndcg@10"]).to_csv(HERE / "per_query.csv", index=False)
    st = pd.DataFrame(stats, columns=["qrels", "dataset", "system", "mean_delta", "p_value", "win", "tie", "loss"])
    st.to_csv(HERE / "paired_tests.csv", index=False)
    augdf = pd.DataFrame(aug_rows, columns=["dataset", "system", "nDCG@10_aug", "J@10_aug"])
    if len(augdf):
        augdf.to_csv(HERE / "results_augmented.csv", index=False)
    pub = published_table(res, dss)
    pub.to_csv(HERE / "published_comparison.csv", index=False)
    (HERE / "eval_info.json").write_text(json.dumps(info, indent=2, default=str))

    plot(res, dss, systems)
    write_examples(examples)
    write_tex(res, st, info, dss, systems, pub, augdf)
    write_result(res, st, info, dss, systems, pub, augdf)
    print((HERE / "RESULT.md").read_text()[:4000])


def published_table(res, dss):
    """Pair each published number (published.py) with our matching system's value.

    Returns a DataFrame with one row per (metric, dataset, published entry): published value, our value
    (None for reference-only entries or missing results) and delta = ours - published.
    """
    out = []
    for kind, table in (("nDCG@10", NDCG10), ("Rcap@100", RECALL100)):
        for ours, label, cite, vals in table:
            for ds in dss:
                pv = vals.get(ds)
                if pv is None:
                    continue
                # Published recall is compared with our capped recall, except the bge model card, which reports uncapped R@100.
                metric = kind if not (kind == "Rcap@100" and ours == "Dense") else "R@100"  # card recall is uncapped
                ov = None
                if ours:
                    m = res[(res.dataset == ds) & (res.system == ours) & (res.metric == metric)]
                    ov = float(m.value.iloc[0]) if len(m) else None
                out.append((kind, ds, label, cite, ours, pv, ov, None if ov is None else ov - pv))
    return pd.DataFrame(out, columns=["metric", "dataset", "published", "cite", "ours_system", "published_value",
                                      "ours_value", "delta"])


def plot(res, dss, systems):
    """Draw one horizontal nDCG@10 bar chart per dataset, colored by family, with a dashed BM25 line.

    Saved as ndcg10.png (project root) and report/ndcg10.pdf.
    """
    v = res[res.metric == "nDCG@10"].pivot(index="system", columns="dataset", values="value")
    fig, axes = plt.subplots(1, len(dss), figsize=(3.4 * len(dss), 0.36 * len(systems) + 1.6), sharey=True)
    # Reverse so the first system in ORDER is drawn at the top.
    ys = np.arange(len(systems))[::-1]
    for ax, ds in zip(np.atleast_1d(axes), dss):
        vals = [v.loc[s, ds] for s in systems]
        ax.barh(ys, vals, height=0.72, color=[FAM_COLOR[FAMILY[s]] for s in systems], edgecolor="#fcfcfb", linewidth=2)
        base = v.loc["BM25", ds]
        ax.axvline(base, color="#52514e", lw=1, ls="--")
        for y, x in zip(ys, vals):
            ax.text(x + 0.008, y, f"{x:.3f}", va="center", fontsize=7, color="#0b0b0b")
        ax.set_title(ds, fontsize=10, color="#0b0b0b")
        # Leave headroom for the value labels to the right of the bars.
        ax.set_xlim(0, min(1.0, max(vals) * 1.22))
        ax.tick_params(labelsize=7, colors="#52514e")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("nDCG@10", fontsize=8, color="#52514e")
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_yticks(ys, systems)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAM_COLOR.values()]
    fig.legend(handles + [plt.Line2D([0], [0], color="#52514e", ls="--")], [*FAM_COLOR, "BM25 baseline"],
               loc="lower center", ncol=8, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(HERE / "ndcg10.png", dpi=200, facecolor="#fcfcfb")
    fig.savefig(HERE / "report" / "ndcg10.pdf", facecolor="#fcfcfb")
    plt.close(fig)


def write_examples(examples):
    """Write examples.md: for each dataset, the queries where Dense most beats/loses to BM25,
    with each system's top 3 docs, their human grade (or 'unjudged') and a text snippet.

    Each entry of examples is the tuple built in main(): (ds, queries, qrels, corpus, runs, per, best, worst, dd).
    """
    out = ["# Dense vs BM25: largest per-query nDCG@10 differences\n"]
    for ds, queries, qrels, corpus, runs, per, best, worst, dd in examples:
        out.append(f"\n## {ds}\n")
        for title, qs in (("Dense beats BM25 most", best), ("BM25 beats Dense most", worst)):
            out.append(f"\n### {title}\n")
            for q in qs:
                out.append(f"\n**qid {q}**: {queries[q]!r}  \nnDCG@10 BM25 = {per['BM25']['nDCG@10'][q]:.3f}, "
                           f"Dense = {per['Dense']['nDCG@10'][q]:.3f} (Δ = {dd[q]:+.3f})\n")
                for s in ("BM25", "Dense"):
                    out.append(f"\n- {s} top 3:")
                    for r, d in enumerate(runs[s].get(q, [])[:3], 1):
                        g = qrels[q].get(d)
                        gs = "unjudged" if g is None else f"grade {g}"
                        # Flatten newlines and replace '|' so the snippet doesn't break the Markdown list or tables.
                        out.append(f"\n  {r}. doc {d} ({gs}): {corpus[d][:200].replace(chr(10), ' ').replace('|', '/')}")
                out.append("\n")
    (HERE / "examples.md").write_text("".join(out))


def tx(s):
    """Escape LaTeX special characters (_ % & #) in a table cell."""
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def write_tex(res, st, info, dss, systems, pub, augdf):
    """Write report/tables.tex: the LaTeX tables included by the report.

    Tables: (1) nDCG@10 with delta vs BM25 and significance stars, (2) secondary metrics,
    (3) comparison with published results, (4) augmented-qrels nDCG@10 (only if augdf is non-empty),
    (5) latency and setup cost. Each uses booktabs rules and is scaled to the line width with resizebox.
    """
    L = []
    nd = res[res.metric == "nDCG@10"].pivot(index="system", columns="dataset", values="value")
    # 1. nDCG@10 matrix with significance vs BM25
    so = st[st.qrels == "original"].set_index(["dataset", "system"])
    L.append(r"\begin{table}[H]\centering\small\caption{nDCG@10 per system (rows) and collection (columns). "
             r"$\Delta$ = difference from BM25; $^{*}$ paired $t$-test $p<0.05$, $^{**}$ $p<0.001$. Best per column in bold.}"
             r"\label{tab:ndcg}" + "\n")
    # Two columns (value, delta) per dataset after System and Family.
    L.append(r"\resizebox{\linewidth}{!}{\begin{tabular}{ll" + "rr" * len(dss) + r"}\toprule" + "\n")
    L.append(" & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{tx(d)}}}" for d in dss) + r"\\" + "\n")
    L.append("System & Family & " + " & ".join(r"nDCG & $\Delta$" for _ in dss) + r"\\\midrule" + "\n")
    for s in systems:
        cells = []
        for ds in dss:
            x = nd.loc[s, ds]
            c = f"{x:.3f}"
            # Bold the best value in each dataset column.
            cells.append(rf"\textbf{{{c}}}" if np.isclose(x, nd[ds].max()) else c)
            if s == "BM25":
                cells.append("--")
            else:
                r = so.loc[(ds, s)]
                # One star for p < ALPHA, two for p < 0.001.
                star = "$^{**}$" if r.p_value < 1e-3 else ("$^{*}$" if r.p_value < ALPHA else "")
                cells.append(f"{r.mean_delta:+.3f}{star}")
        L.append(f"{tx(s)} & {FAMILY[s]} & " + " & ".join(cells) + r"\\" + "\n")
    L.append(r"\bottomrule\end{tabular}}\end{table}" + "\n\n")
    # 2. secondary metrics
    sec = ["R@100", "Rcap@100", "P@10", "AP@100", "J@10"]
    L.append(r"\begin{table}[H]\centering\scriptsize\caption{Secondary metrics. R$_{cap}$@100 divides by "
             r"$\min(|rel|,100)$ as in BEIR Table 9; J@10 is a diagnostic (share of top-10 docs with any human "
             r"judgment).}\label{tab:sec}" + "\n")
    L.append(r"\resizebox{\linewidth}{!}{\begin{tabular}{l" + "r" * (len(sec) * len(dss)) + r"}\toprule" + "\n")
    L.append(" & " + " & ".join(rf"\multicolumn{{{len(sec)}}}{{c}}{{{tx(d)}}}" for d in dss) + r"\\" + "\n")
    L.append("System & " + " & ".join(" & ".join(["R@100", r"R$_{cap}$", "P@10", "AP", "J@10"]) for _ in dss) + r"\\\midrule" + "\n")
    for s in systems:
        cells = []
        for ds in dss:
            g = res[(res.dataset == ds) & (res.system == s)].set_index("metric").value
            cells += [f"{g[m]:.3f}" for m in sec]
        L.append(f"{tx(s)} & " + " & ".join(cells) + r"\\" + "\n")
    L.append(r"\bottomrule\end{tabular}}\end{table}" + "\n\n")
    # 3. published comparison
    L.append(r"\begin{table}[H]\centering\scriptsize\caption{Our runs vs.\ published results (nDCG@10 unless noted). "
             r"``Ours'' is the matching system from this study; $\Delta$ = ours $-$ published. "
             r"Rows without a match are reference points only.}\label{tab:pub}" + "\n")
    L.append(r"\resizebox{\linewidth}{!}{\begin{tabular}{p{5.2cm}l" + "rr" * len(dss) + r"}\toprule" + "\n")
    L.append(" & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{tx(d)}}}" for d in dss) + r"\\" + "\n")
    L.append("Published (source) & Ours & " + " & ".join(r"pub. & $\Delta$" for _ in dss) + r"\\\midrule" + "\n")
    for kind in ("nDCG@10", "Rcap@100"):
        sub = pub[pub.metric == kind]
        if kind == "Rcap@100":
            L.append(r"\midrule\multicolumn{" + str(2 + 2 * len(dss)) + r"}{l}{\emph{Recall@100}}\\" + "\n")
        # Unique published labels in first-seen order (dict.fromkeys keeps insertion order).
        for label in dict.fromkeys(sub.published):
            r0 = sub[sub.published == label]
            cite = r0.cite.iloc[0]
            # ours_system is NaN (not a str) for reference-only rows.
            ours = r0.ours_system.iloc[0] if isinstance(r0.ours_system.iloc[0], str) else "--"
            cells = []
            for ds in dss:
                r = r0[r0.dataset == ds]
                if not len(r):
                    cells += ["--", ""]
                else:
                    r = r.iloc[0]
                    cells += [f"{r.published_value:.3f}", "" if pd.isna(r.delta) else f"{r.delta:+.3f}"]
            L.append(rf"{tx(label)}~\cite{{{cite}}} & {tx(ours)} & " + " & ".join(cells) + r"\\" + "\n")
    L.append(r"\bottomrule\end{tabular}}\end{table}" + "\n\n")
    # 4. augmented qrels
    if len(augdf):
        sa = st[st.qrels == "augmented"].set_index(["dataset", "system"])
        L.append(r"\begin{table}[H]\centering\small\caption{nDCG@10 with LLM labels filling the judgment holes "
                 r"(Thomas et al.~\cite{thomas2024llmjudge} prompt, Qwen3.5-122B judge). $\Delta$ vs.\ BM25 under the "
                 r"same augmented qrels; $^{*}$ $p<0.05$, $^{**}$ $p<0.001$.}\label{tab:aug}" + "\n")
        L.append(r"\resizebox{\linewidth}{!}{\begin{tabular}{l" + "rrr" * len(dss) + r"}\toprule" + "\n")
        L.append(" & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{tx(d)}}}" for d in dss) + r"\\" + "\n")
        L.append("System & " + " & ".join(r"orig. & aug. & $\Delta$" for _ in dss) + r"\\\midrule" + "\n")
        a = augdf.set_index(["dataset", "system"])
        for s in systems:
            cells = []
            for ds in dss:
                cells += [f"{nd.loc[s, ds]:.3f}", f"{a.loc[(ds, s), 'nDCG@10_aug']:.3f}"]
                if s == "BM25":
                    cells.append("--")
                else:
                    r = sa.loc[(ds, s)]
                    star = "$^{**}$" if r.p_value < 1e-3 else ("$^{*}$" if r.p_value < ALPHA else "")
                    cells.append(f"{r.mean_delta:+.3f}{star}")
            L.append(f"{tx(s)} & " + " & ".join(cells) + r"\\" + "\n")
        L.append(r"\bottomrule\end{tabular}}\end{table}" + "\n\n")
    # 5. latency
    L.append(r"\begin{table}[H]\centering\small\caption{Mean end-to-end latency per query (ms; model load excluded; "
             r"LLM methods include the gateway call time) and one-time index/encode cost (s).}\label{tab:lat}" + "\n")
    L.append(r"\resizebox{\linewidth}{!}{\begin{tabular}{l" + "rr" * len(dss) + r"}\toprule" + "\n")
    L.append(" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{tx(d)}}}" for d in dss) + r"\\" + "\n")
    L.append("System & " + " & ".join("ms/q & setup s" for _ in dss) + r"\\\midrule" + "\n")
    for s in systems:
        cells = []
        for ds in dss:
            g = res[(res.dataset == ds) & (res.system == s)].set_index("metric").value
            su = g["setup_s"]
            cells += [f"{g['latency_ms']:,.1f}", "--" if su is None or pd.isna(su) else f"{su:,.0f}"]
        L.append(f"{tx(s)} & " + " & ".join(cells) + r"\\" + "\n")
    L.append(r"\bottomrule\end{tabular}}\end{table}" + "\n")
    (HERE / "report" / "tables.tex").write_text("".join(L))


def versions():
    """Return installed versions of the key packages for the Environment section ('n/a' if missing)."""
    pk = ["bm25s", "PyStemmer", "sentence-transformers", "transformers", "torch", "ir-measures",
          "pytrec-eval-terrier", "numpy", "pandas", "scipy", "matplotlib", "openai"]
    out = {}
    for p in pk:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = "n/a"
    return out


def cpu():
    """Return the CPU model string (macOS sysctl, falling back to platform.processor())."""
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return platform.processor()


def write_result(res, st, info, dss, systems, pub, augdf):
    """Write RESULT.md: verdict per dataset, metric tables, published comparison, augmented-qrels results
    and judge validation, paired tests, latency, per-system configuration and environment details.
    """
    M = ["nDCG@10", "R@100", "Rcap@100", "P@10", "AP@100", "J@10"]
    L = ["# SPIKE T1 RESULT: semantic retrieval vs BM25 and vs published results (" + ", ".join(dss) + ")\n\n"]
    L.append("## VERDICT (nDCG@10; paired t-test vs BM25, alpha = 0.05, original qrels)\n\n")
    so = st[st.qrels == "original"]
    for ds in dss:
        parts = []
        for _, r in so[so.dataset == ds].iterrows():
            # Call a difference 'beats'/'loses to' only when significant; otherwise 'n.s.'.
            v = ("beats" if r.mean_delta > 0 else "loses to") if r.p_value < ALPHA else "n.s. vs"
            parts.append(f"{r.system} {v} BM25 ({r.mean_delta:+.3f}, p={fmt_p(r.p_value)})")
        L.append(f"- **{ds}**: " + "; ".join(parts) + ".\n")
    L.append("\n## Results (systems x metrics)\n")
    for ds in dss:
        sub = res[res.dataset == ds].pivot(index="system", columns="metric", values="value").loc[systems]
        L.append(f"\n### {ds} ({info[ds]['n_eval']} evaluated queries)\n\n| System | " + " | ".join(M) +
                 " |\n|---|" + "---:|" * len(M) + "\n")
        for s in systems:
            L.append(f"| {s} | " + " | ".join(f"{sub.loc[s, m]:.4f}" for m in M) + " |\n")
    L.append("\n## Published comparison (ours - published)\n\n| Metric | Dataset | Published (source) | Ours | Published | Ours value | Delta |\n|---|---|---|---|---:|---:|---:|\n")
    for _, r in pub.iterrows():
        ov = "" if r.ours_value is None or pd.isna(r.ours_value) else f"{r.ours_value:.4f}"
        de = "" if r.delta is None or pd.isna(r.delta) else f"{r.delta:+.4f}"
        L.append(f"| {r.metric} | {r.dataset} | {r.published} [{r.cite}] | {r.ours_system or '-'} | {r.published_value:.4f} | {ov} | {de} |\n")
    if len(augdf):
        L.append("\n## Augmented qrels (LLM labels for holes)\n\n| Dataset | System | nDCG@10 orig | nDCG@10 aug | J@10 aug |\n|---|---|---:|---:|---:|\n")
        nd = res[res.metric == "nDCG@10"].set_index(["dataset", "system"]).value
        for _, r in augdf.iterrows():
            L.append(f"| {r.dataset} | {r.system} | {nd[(r.dataset, r.system)]:.4f} | {r['nDCG@10_aug']:.4f} | {r['J@10_aug']:.3f} |\n")
        vf = HERE / "llm_labels" / "validation.json"
        if vf.exists():
            L.append("\nJudge validation vs human labels:\n\n```\n" + vf.read_text() + "\n```\n")
    L.append("\n## Paired tests vs BM25 (per-query nDCG@10; tie = |delta| < 0.01)\n\n| Qrels | Dataset | System | Mean delta | p | Win | Tie | Loss |\n|---|---|---|---:|---:|---:|---:|---:|\n")
    for _, r in st.iterrows():
        L.append(f"| {r.qrels} | {r.dataset} | {r.system} | {r.mean_delta:+.4f} | {fmt_p(r.p_value)} | {r.win} | {r.tie} | {r.loss} |\n")
    L.append("\n## Latency and setup\n\n| Dataset | System | ms/query | setup s |\n|---|---|---:|---:|\n")
    for ds in dss:
        for s in systems:
            g = res[(res.dataset == ds) & (res.system == s)].set_index("metric").value
            L.append(f"| {ds} | {s} | {g['latency_ms']:.1f} | {g['setup_s'] if pd.isna(g['setup_s']) else round(g['setup_s'], 1)} |\n")
    L.append("\n## Configuration per system (timing/*.json)\n\n")
    for s in systems:
        # Configuration is read from the first dataset's timing file; LLM usage stats are too verbose to list.
        t = dict(info[dss[0]]["timing"][s])
        t.pop("llm_stats", None)
        L.append(f"- **{s}**: " + ", ".join(f"{k}={v}" for k, v in t.items() if k not in ("system", "mean_ms")) + "\n")
    L.append("\n## Environment\n\n")
    L.append(f"- OS: {platform.platform()}; Python {platform.python_version()}; CPU/GPU: {cpu()} (MPS, no CUDA)\n")
    L.append("- Libraries (.venv): " + ", ".join(f"{k} {v}" for k, v in versions().items()) + "\n")
    L.append("- ColBERTv2 runs in .venv-colbert (PyLate pins torch/transformers/sentence-transformers older versions).\n")
    L.append("- Seeds: 0 (random, numpy, torch). LLM calls cached in llm_cache/ (reruns replay the cache).\n")
    f = HERE / "step0_probe_beir.txt"
    if f.exists():
        L.append("\n## Raw Step 0 probe output\n\n```\n" + f.read_text() + "```\n")
    (HERE / "RESULT.md").write_text("".join(L))


if __name__ == "__main__":
    main()

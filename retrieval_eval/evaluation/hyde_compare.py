"""Compare HyDE with the gpt-oss-120b generator (original runs) against the Llama 3.3 70B generator.

For each encoder (bge, Contriever, Contriever with 100-word passages) and dataset: nDCG@10 and
R@100 for both generators, the paired t-test of Llama vs. gpt-oss on per-query nDCG@10, the
no-HyDE baseline (Dense / Contriever) and the published HyDE number, plus passage statistics
(median words, share with markdown) and per-query latency. Reads runs/, timing/ and llm_cache/
only; writes hyde_llama_comparison.csv and hyde_llama_comparison.md.

    .venv/bin/python -m retrieval_eval.evaluation.hyde_compare
"""
import json
import re
import statistics

import pandas as pd
from scipy.stats import ttest_rel

from retrieval_eval.common import BEIR_SETS, HERE, load, read_run, read_timing
from retrieval_eval.evaluation.evaluate import evaluate
from retrieval_eval.methods.hyde import GENERATORS, STOP

# (label, gpt-oss system, Llama system, no-HyDE baseline system)
PAIRS = [("bge", "HyDE-bge", "HyDE-bge-llama", "Dense"),
         ("Contriever", "HyDE-Contriever", "HyDE-Contriever-llama", "Contriever"),
         ("Contriever-100w", "HyDE-Contriever-100w", "HyDE-Contriever-100w-llama", "Contriever")]
PUBLISHED = {"scifact": 0.691, "fiqa": 0.273, "trec-covid": 0.593}  # HyDE Table 2 (Contriever encoder)
MARKDOWN = re.compile(r"^\s*(#|\||\*\*|[-*] )", re.M)  # headings, table rows, bold, bullet lists


def passage_stats(gen, ds):
    """Median word count and share of passages containing markdown, for one generator and dataset."""
    words, md = [], 0
    with open(HERE / "llm_cache" / f"{gen['model']}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("tag", "").startswith(f"{gen['tag']}:{ds}:"):
                t = r["text"].split(STOP)[0].strip()
                words.append(len(t.split()))
                md += bool(MARKDOWN.search(t))
    return (statistics.median(words), md / len(words)) if words else (float("nan"), float("nan"))


def md_table(df):
    """Render a DataFrame as a GitHub markdown table (avoids the optional tabulate dependency)."""
    cells = lambda r: "| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |"
    return "\n".join([cells(df.columns), "|" + "---|" * len(df.columns)] + [cells(r) for r in df.itertuples(index=False)])


def main():
    """Score both generators' HyDE runs on every dataset and write the comparison table."""
    rows = []
    for ds in BEIR_SETS:
        _, queries, qrels = load(ds)
        evq = [q for q in queries if q in qrels and any(g > 0 for g in qrels[q].values())]
        systems = {s for p in PAIRS for s in p[1:]}
        per = evaluate(qrels, {s: read_run(ds, s) for s in systems}, evq)
        mean = lambda s, m: sum(per[s][m].values()) / len(evq)
        pw = {k: passage_stats(GENERATORS[k], ds) for k in GENERATORS}
        for label, old, new, base in PAIRS:
            a = [per[old]["nDCG@10"][q] for q in evq]
            b = [per[new]["nDCG@10"][q] for q in evq]
            d = [y - x for x, y in zip(a, b)]
            rows.append({
                "dataset": ds, "encoder": label,
                "baseline (no HyDE)": round(mean(base, "nDCG@10"), 3),
                "gpt-oss nDCG@10": round(mean(old, "nDCG@10"), 3),
                "llama nDCG@10": round(mean(new, "nDCG@10"), 3),
                "delta": round(sum(d) / len(d), 3),
                "p (paired t)": ttest_rel(b, a).pvalue if any(d) else 1.0,
                "llama wins/losses": f"{sum(x >= 0.01 for x in d)}/{sum(x <= -0.01 for x in d)}",
                "published (Contriever)": PUBLISHED.get(ds) if label == "Contriever" else None,
                "gpt-oss R@100": round(mean(old, "R@100"), 3),
                "llama R@100": round(mean(new, "R@100"), 3),
                "gpt-oss ms/q": round(read_timing(ds, old)["mean_ms"]),
                "llama ms/q": round(read_timing(ds, new)["mean_ms"]),
                "gpt-oss words": pw["gpt-oss"][0], "llama words": pw["llama"][0],
                "gpt-oss markdown %": round(100 * pw["gpt-oss"][1]), "llama markdown %": round(100 * pw["llama"][1]),
            })
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "hyde_llama_comparison.csv", index=False)
    show = df.copy()
    show["p (paired t)"] = show["p (paired t)"].map(lambda p: f"{p:.2g}")
    (HERE / "hyde_llama_comparison.md").write_text(
        "# HyDE: gpt-oss-120b vs Llama 3.3 70B generator\n\n"
        "Same prompts, 8 samples, temperature 0.7; nDCG@10 on the original BEIR qrels; delta = Llama - gpt-oss; "
        "paired t-test on per-query nDCG@10 (no multiple-comparison correction).\n\n" + md_table(show) + "\n")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()

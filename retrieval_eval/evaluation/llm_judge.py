"""LLM relevance labels for judgment holes (Thomas et al., SIGIR 2024).

Prompt: the paper's Figure 2 prompt (role, 0-2 scale and its definitions, "A person has typed
[query] into a search engine", web-page block, aspects M/T/O, JSON output "without providing
any reasoning"), single judge (no "multiple" variant), no description/narrative (BEIR queries
have no separate description). Judge: Qwen3.5-122B-A10B with thinking disabled
(reasoning effort none), temperature 0 - deliberately not the gpt-oss model used by RankGPT/HyDE.

--validate : agreement with human labels before any hole is filled.
    TREC-COVID has graded 0/1/2 human labels -> Cohen's kappa (binary rel>=1, and 3-class
    linear-weighted) on a stratified sample. SciFact/NFCorpus/FiQA qrels contain only relevant
    docs, so kappa is undefined there; we report the share of known-relevant docs the LLM labels >=1.
--holes    : label every unjudged (query, doc) pair in any system's top 10 (all runs/ files),
    write qrels_aug/{ds}.tsv = human qrels + LLM labels for holes (human labels never overridden).
    Binary-judged sets (SciFact, FiQA) get O>=1 -> 1; graded sets keep 0-2.
"""
import json
import random
import re
from collections import Counter

from retrieval_eval.common import BEIR_SETS, HERE, RUNS, cli, load, read_run
from retrieval_eval.llm import QWEN, LLM

# Deterministic, short JSON answer; thinking off so the output is just the score array.
PARAMS = {"temperature": 0, "max_output_tokens": 64, "reasoning": {"effort": "none"}}
# Documents are truncated to this many whitespace tokens before prompting.
MAX_WORDS = 512
# Pairs sampled per dataset for --validate.
N_VALIDATE = 200
# Datasets whose human qrels are binary; LLM grades are collapsed to 0/1 for them.
BINARY = {"scifact", "fiqa"}
OUT = HERE / "qrels_aug"
LABELS = HERE / "llm_labels"

# Thomas et al. (2024) Figure 2 prompt; literal JSON braces are doubled for str.format.
PROMPT = """You are a search quality rater evaluating the relevance of web pages. Given a query and a web page, you must provide a score on an integer scale of 0 to 2 with the following meanings:
2 = highly relevant, very helpful for this query
1 = relevant, may be partly helpful but might contain other irrelevant content
0 = not relevant, should never be shown for this query
Assume that you are writing a report on the subject of the topic. If you would use any of the information contained in the web page in such a report, mark it 1. If the web page is primarily about the topic, or contains vital information about the topic, mark it 2. Otherwise, mark it 0.

Query
A person has typed [{query}] into a search engine.

Result
Consider the following web page.
—BEGIN WEB PAGE CONTENT—
{page}
—END WEB PAGE CONTENT—

Instructions
Split this problem into steps:
Consider the underlying intent of the search.
Measure how well the content matches a likely intent of the query (M).
Measure how trustworthy the web page is (T).
Consider the aspects above and the relative importance of each, and decide on a final score (O).
Produce a JSON array of scores without providing any reasoning. Example: [{{"M": 2, "T": 1, "O": 1}}]

Results
"""


def opts(ap):
    """Add the required, mutually exclusive --validate / --holes mode flags to the argparse parser."""
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate", action="store_true")
    g.add_argument("--holes", action="store_true")


def parse(text):
    """Extract the overall score O (0-2) from the judge's JSON output; None if it can't be found.

    A regex rather than json.loads tolerates extra text or malformed JSON around the array.
    """
    m = re.search(r'"O"\s*:\s*([0-2])', text or "")
    return int(m.group(1)) if m else None


def judge(llm, ds, pairs, queries, corpus, desc):
    """Label (query, doc) pairs with the LLM judge, in parallel and cached.

    Returns (labels, raw_outputs): labels are parsed O scores (None when unparsable), in pair order.
    The cache tag judge:{ds}:{q}:{d} identifies each call in llm_cache/.
    """
    jobs = [(PROMPT.format(query=queries[q], page=" ".join(corpus[d].split()[:MAX_WORDS])),
             f"judge:{ds}:{q}:{d}", PARAMS) for q, d in pairs]
    outs = llm.map(jobs, desc=desc)
    return [parse(o) for o in outs], outs


def kappa(a, b, weighted=False, k=3):
    """Cohen's kappa between label lists a and b over classes 0..k-1.

    Written as 1 - observed/expected disagreement, with disagreement weight w(i, j): 0/1 for plain
    kappa, |i - j| / (k - 1) for linear-weighted kappa. Expected disagreement uses each rater's
    marginal label counts (chance agreement). Returns NaN for empty input or zero expected disagreement.
    """
    n = len(a)
    if n == 0:
        return float("nan")
    w = (lambda i, j: abs(i - j) / (k - 1)) if weighted else (lambda i, j: float(i != j))
    obs = sum(w(x, y) for x, y in zip(a, b)) / n
    pa, pb = Counter(a), Counter(b)
    exp = sum(pa[i] * pb[j] * w(i, j) for i in range(k) for j in range(k)) / (n * n)
    return 1 - obs / exp if exp else float("nan")


def validate(llm, datasets):
    """--validate: compare LLM labels with human labels on a sample of judged pairs per dataset.

    Writes llm_labels/validation.json (also embedded in RESULT.md by evaluate.py) and returns the dict.
    """
    rng = random.Random(0)
    # Fixed seed so the validation sample is reproducible.
    LABELS.mkdir(exist_ok=True)
    res = {}
    for ds in datasets:
        corpus, queries, qrels = load(ds)
        judged = [(q, d, g) for q, v in qrels.items() for d, g in v.items()]
        if ds == "trec-covid":  # stratified by human grade
            by = {g: [x for x in judged if x[2] == g] for g in (0, 1, 2)}
            # Equal share per grade 0/1/2; the remainder of N_VALIDATE / 3 goes to grade 0.
            sample = [x for g in (0, 1, 2) for x in rng.sample(by[g], N_VALIDATE // 3 + (g == 0) * (N_VALIDATE % 3))]
        else:
            sample = rng.sample(judged, min(N_VALIDATE, len(judged)))
        labels, _ = judge(llm, ds, [(q, d) for q, d, _ in sample], queries, corpus, f"validate {ds}")
        # Drop pairs whose output couldn't be parsed; their count is reported as 'unparsed'.
        ok = [(g, l) for (_, _, g), l in zip(sample, labels) if l is not None]
        h, m = [g for g, _ in ok], [l for _, l in ok]
        r = {"n": len(sample), "unparsed": len(sample) - len(ok),
             "llm_label_dist": dict(Counter(m)), "human_label_dist": dict(Counter(h))}
        if ds == "trec-covid":
            # Graded human labels: kappa on binary relevance (grade >= 1) and linear-weighted 3-class kappa.
            r["kappa_binary"] = kappa([int(x >= 1) for x in h], [int(x >= 1) for x in m], k=2)
            r["kappa_3class_linear"] = kappa(h, m, weighted=True)
            r["exact_agreement"] = sum(x == y for x, y in zip(h, m)) / len(ok)
        else:
            # Qrels hold only relevant docs, so kappa is undefined; report recall of known-relevant docs instead.
            r["share_known_relevant_labelled_rel"] = sum(x >= 1 for x in m) / len(ok)
        res[ds] = r
        print(f"[judge-validate] {ds}: {json.dumps(r)}", flush=True)
    (LABELS / "validation.json").write_text(json.dumps(res, indent=2))
    return res


def holes(llm, datasets):
    """--holes: LLM-label every unjudged doc in any system's top 10 and write augmented qrels.

    Writes llm_labels/{ds}.jsonl (raw judge outputs) and qrels_aug/{ds}.tsv (human qrels first, then
    LLM labels for holes; unparsable labels are left out, so those pairs stay unjudged).
    """
    OUT.mkdir(exist_ok=True)
    LABELS.mkdir(exist_ok=True)
    for ds in datasets:
        corpus, queries, qrels = load(ds)
        systems = sorted({p.name[len(ds) + 1:-len(".trec")] for p in RUNS.glob(f"{ds}.*.trec")})
        # Union of unjudged (query, doc) pairs in the top 10 of every run on disk, deduplicated.
        pairs = sorted({(q, d) for s in systems for q, docs in read_run(ds, s).items() if q in qrels
                        for d in docs[:10] if d not in qrels[q]})
        print(f"[judge-holes] {ds}: {len(pairs)} unjudged top-10 pairs across {len(systems)} systems", flush=True)
        labels, outs = judge(llm, ds, pairs, queries, corpus, f"holes {ds}")
        with open(LABELS / f"{ds}.jsonl", "w") as f:
            for (q, d), l, o in zip(pairs, labels, outs):
                f.write(json.dumps({"qid": q, "doc": d, "O": l, "raw": o[:200]}) + "\n")
        with open(OUT / f"{ds}.tsv", "w") as f:
            f.write("query-id\tcorpus-id\tscore\n")
            for q, v in qrels.items():
                for d, g in v.items():
                    f.write(f"{q}\t{d}\t{g}\n")
            for (q, d), l in zip(pairs, labels):
                # Collapse grade 2 to 1 on binary-judged sets so the scale matches the human labels.
                if l is not None:
                    f.write(f"{q}\t{d}\t{min(l, 1) if ds in BINARY else l}\n")
        dist = Counter(labels)
        print(f"[judge-holes] {ds}: labels {dict(dist)} (None = unparsed, left unjudged)", flush=True)


def main():
    """Run --validate or --holes on the selected datasets with the Qwen judge, then print LLM usage stats."""
    args = cli(__doc__, opts)
    llm = LLM(QWEN)
    if args.validate:
        validate(llm, args.datasets)
    else:
        holes(llm, args.datasets)
    llm.report("judge")


if __name__ == "__main__":
    main()

"""RankGPT: listwise LLM reranking by permutation generation (Sun et al., EMNLP 2023).

Port of the official sunnweiwei/RankGPT rank_gpt.py: exact prompts (get_prefix_prompt /
get_post_prompt, "Received passage [i]." turns), passage text "Title: ... Content: ..."
whitespace-normalised and cut to 300 words, temperature 0, sliding window back-to-first with
window 20 / step 10 over the top 100 (9 windows), and receive_permutation(): keep digits,
drop duplicates and out-of-range ids, append missing ids in original order.
First stage: BM25-BEIR top 100 (the paper reranks "the same BM25 top-100 passages").
LLM: gpt-oss-120b (reasoning effort low) via Databricks Unity Gateway; the paper used
gpt-3.5-turbo / gpt-4. Published nDCG@10 (RankGPT Table 1), Covid / NFCorpus / SciFact:
BM25 59.47 / 30.75 / 67.89; gpt-3.5-turbo 76.67 / 35.62 / 70.43; gpt-4 85.51 / 38.47 / 74.95.
FiQA is not in the paper.
"""
import time

from retrieval_eval.common import cli, load_beir_fields, load, read_run, read_timing, write_run, write_timing
from retrieval_eval.llm import GPT_OSS, LLM

WINDOW, STEP, DEPTH, MAX_WORDS = 20, 10, 100, 300
SOURCE, NAME = "BM25-BEIR", "RankGPT"
PARAMS = {"temperature": 0, "max_output_tokens": 1024, "reasoning": {"effort": "low"}}


def prefix(query, num):
    """Opening chat turns of the official RankGPT prompt (system + task intro + assistant ack)."""
    return [{"role": "system",
             "content": "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."},
            {"role": "user",
             "content": f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."},
            {"role": "assistant", "content": "Okay, please provide the passages."}]


def post(query, num):
    """Final user instruction asking for the ranking as "[i] > [j] > ..." (verbatim, including its typos)."""
    return (f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. "
            "The passages should be listed in descending order using identifiers. The most relevant passages should be "
            "listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, "
            "do not say any word or explain.")


def messages(query, passages):
    """Build the full chat for one window: prefix, then one user/assistant turn pair per passage, then post."""
    m = prefix(query, len(passages))
    # Each passage is its own turn, numbered [1]..[n] (window-local ids), acknowledged by the assistant.
    for i, p in enumerate(passages, 1):
        m.append({"role": "user", "content": f"[{i}] {p}"})
        m.append({"role": "assistant", "content": f"Received passage [{i}]."})
    m.append({"role": "user", "content": post(query, len(passages))})
    return m


def receive(response, n):
    """Official receive_permutation(): returns (order, n_dup, n_missing).

    Parse an LLM ranking like "[3] > [1] > [2]" over n passages into a full permutation of
    0-based window positions. Robust to malformed output: any non-digit is a separator,
    repeats keep their first occurrence, out-of-range ids are dropped, and ids the model
    omitted are appended in their original order (so an empty response leaves order unchanged).
    """
    digits = "".join(c if c.isdigit() else " " for c in response).split()
    raw = [int(x) - 1 for x in digits]  # prompt ids are 1-based
    seen, order = set(), []
    for x in raw:
        if x not in seen:
            seen.add(x)
            order.append(x)
    order = [x for x in order if 0 <= x < n]
    missing = [x for x in range(n) if x not in order]
    return order + missing, len(raw) - len(set(raw)), len(missing)


def main():
    """Rerank the BM25-BEIR top 100 per query with sliding-window LLM permutation, write run/timing."""
    args = cli(__doc__)
    llm = LLM(GPT_OSS)
    for ds in args.datasets:
        _, queries, _ = load(ds)
        fields = load_beir_fields(ds)
        # official order: whitespace-normalise, drop an empty "Title: Content: " prefix, cut to 300 words
        text = {d: " ".join(" ".join(f"Title: {t} Content: {b}".split()).replace("Title: Content: ", "").strip()
                            .split()[:MAX_WORDS]) for d, (t, b) in fields.items()}
        src = read_run(ds, SOURCE)
        # Windows within a query are sequential (each depends on the previous); queries run in parallel.
        state = {q: list(src.get(q, []))[:DEPTH] for q in queries}
        stats = {"windows": 0, "dup": 0, "missing": 0, "empty": 0}
        qtime = {q: 0.0 for q in queries}
        # Sliding window from the bottom up: [80,100), [70,90), ..., [0,20). Windows overlap by
        # WINDOW - STEP, so a relevant passage can be carried upward across windows toward the top.
        # One window position is processed for all queries at once (one parallel llm.map batch).
        end = DEPTH
        while end - WINDOW >= 0:
            start = end - WINDOW
            jobs, keys = [], []
            for q in queries:
                win = state[q][start:end]
                if len(win) < 2:  # nothing to reorder (short first-stage list)
                    continue
                jobs.append((messages(queries[q], [text[d] for d in win]), f"rankgpt:{ds}:{q}:{start}", PARAMS))
                keys.append(q)
            outs = llm.map(jobs, desc=f"RankGPT {ds} window {start}-{end}")
            for q, out in zip(keys, outs):
                win = state[q][start:end]
                order, dup, miss = receive(out, len(win))
                state[q][start:end] = [win[j] for j in order]  # write the permuted window back in place
                stats["windows"] += 1; stats["dup"] += dup; stats["missing"] += miss; stats["empty"] += not out
            end -= STEP
        # Per-query LLM time = sum of its window calls (they are sequential), from recorded call durations.
        for r in llm.cache.values():
            tag = r.get("tag", "")
            if tag.startswith(f"rankgpt:{ds}:"):
                qtime[tag.split(":")[2]] += r["seconds"]
        run = {q: state[q] + list(src.get(q, []))[DEPTH:] for q in queries}  # ranks past DEPTH keep first-stage order
        write_run(ds, NAME, run)
        rr_ms = 1000 * sum(qtime.values()) / len(qtime)
        total = read_timing(ds, SOURCE)["mean_ms"] + rr_ms
        write_timing(ds, NAME, {"system": NAME, "rerank_ms": rr_ms, "mean_ms": total,
                                "setup_s": read_timing(ds, SOURCE).get("setup_s"), "source": SOURCE,
                                "n_queries": len(queries), "llm": GPT_OSS, "window": WINDOW, "step": STEP,
                                "depth": DEPTH, "max_words": MAX_WORDS, **stats,
                                "note": "rerank_ms = sum of the 9 sequential window calls per query"})
        print(f"[{NAME}] {ds}: {stats['windows']} windows, dup ids {stats['dup']}, missing ids {stats['missing']}, "
              f"empty responses {stats['empty']}; {rr_ms / 1000:.1f} s/q LLM time", flush=True)
    llm.report("RankGPT")


if __name__ == "__main__":
    main()

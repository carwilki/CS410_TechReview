"""End-to-end entry point: python run.py [--datasets ...] [--no-llm]

Each step runs one module as `python -m retrieval_eval.<package>.<module>` from the project root.

Order: lexical -> dense -> fusion -> cross-encoder rerankers -> SPLADE -> ColBERTv2 (own venv,
PyLate pins older torch) -> HyDE -> RankGPT -> LLM judge (validate, holes) -> evaluate.
LLM steps hit the Databricks gateway; every call is cached in llm_cache/, so a rerun with a warm
cache makes no gateway calls. --no-llm skips HyDE/RankGPT/judge (evaluate uses whatever runs exist).
Any other arguments (e.g. --datasets scifact) are forwarded unchanged to every step.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# ColBERTv2 runs under a separate interpreter because PyLate needs an older torch than the main venv.
PY, PY_COLBERT = sys.executable, str(HERE / ".venv-colbert/bin/python")
# Steps that run fully offline, as (interpreter, [module, *module_args]); order matters because
# later steps (hybrid, rerank) read the run files written by earlier ones.
LOCAL = [
    (PY, ["retrieval_eval.methods.bm25"]),
    (PY, ["retrieval_eval.methods.bm25_beir"]),
    (PY, ["retrieval_eval.methods.dense"]),
    (PY, ["retrieval_eval.methods.hybrid"]),
    (PY, ["retrieval_eval.methods.rerank"]),                                                   # Hybrid top 50 -> Rerank
    (PY, ["retrieval_eval.methods.rerank", "--source", "BM25-BEIR", "--depth", "100", "--name", "BM25+CE"]),  # BEIR setup
    (PY, ["retrieval_eval.methods.splade"]),
    (PY_COLBERT, ["retrieval_eval.methods.colbert"]),
]
# Steps that call the LLM gateway (cached in llm_cache/); skipped with --no-llm.
LLM_STEPS = [
    (PY, ["retrieval_eval.methods.hyde"]),
    (PY, ["retrieval_eval.methods.rankgpt"]),
    (PY, ["retrieval_eval.evaluation.llm_judge", "--validate"]),
    (PY, ["retrieval_eval.evaluation.llm_judge", "--holes"]),   # must run after every system has written its run
]

if __name__ == "__main__":
    args = sys.argv[1:]
    no_llm = "--no-llm" in args
    extra = [a for a in args if a != "--no-llm"]  # everything else is passed through to each step
    (HERE / "report").mkdir(exist_ok=True)
    t0 = time.perf_counter()
    for py, step in LOCAL + ([] if no_llm else LLM_STEPS):
        print(f"=== {' '.join(step)}", flush=True)
        # check=True aborts the pipeline on the first failing step.
        subprocess.run([py, "-m", step[0], *step[1:], *extra], check=True, cwd=HERE)
    # Record disk used by venvs, indexes, embedding cache and HF model downloads (for cleanup).
    hf = Path.home() / ".cache/huggingface/hub"
    du = subprocess.run(["du", "-sh", str(HERE / ".venv"), str(HERE / ".venv-colbert"), str(HERE / "indexes"),
                         str(HERE / "emb_cache"), str(hf)], capture_output=True, text=True).stdout
    (HERE / "teardown.txt").write_text(du)
    # Final scoring over every run file present in runs/.
    print("=== retrieval_eval.evaluation.evaluate", flush=True)
    subprocess.run([PY, "-m", "retrieval_eval.evaluation.evaluate", *extra], check=True, cwd=HERE)
    if not no_llm:
        # HyDE vector-geometry figures and table for the report (reads the cached generations only).
        print("=== retrieval_eval.evaluation.hyde_geometry", flush=True)
        subprocess.run([PY, "-m", "retrieval_eval.evaluation.hyde_geometry"], check=True, cwd=HERE)
    print(f"=== total wall clock {time.perf_counter() - t0:.0f}s\n=== disk usage (delete manually):\n{du}")

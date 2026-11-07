"""Published nDCG@10 (BEIR test splits) used for comparison. Every value was copied from the
cited table by the source-review agents (extracted paper texts in the session scratchpad).
None = not reported. Values on a 0-1 scale.

Each entry: (our matching system or None for reference-only, source label, citation key, values).
"""
DS = ["scifact", "nfcorpus", "fiqa", "trec-covid"]

# nDCG@10 rows. "T2" etc. refer to the table number in the cited paper.
NDCG10 = [
    ("BM25-BEIR", "BM25, Anserini k1=0.9 b=0.4 (BEIR T2)", "thakur2021beir",
     {"scifact": 0.665, "nfcorpus": 0.325, "fiqa": 0.236, "trec-covid": 0.656}),
    ("BM25+CE", "BM25 top-100 + MiniLM-L6 CE (BEIR T2)", "thakur2021beir",
     {"scifact": 0.688, "nfcorpus": 0.350, "fiqa": 0.347, "trec-covid": 0.757}),
    ("Dense", "bge-small-en-v1.5 (model card, MTEB)", "bge2023card",
     {"scifact": 0.71275, "nfcorpus": 0.34305, "fiqa": 0.40343, "trec-covid": 0.75899}),
    ("SPLADE", "DistilSPLADE-max (SPLADE v2 T2)", "formal2021spladev2",
     {"scifact": 0.693, "nfcorpus": 0.334, "fiqa": 0.336, "trec-covid": 0.710}),
    ("ColBERTv2", "ColBERTv2 (ColBERTv2 T5a)", "santhanam2022colbertv2",
     {"scifact": 0.693, "nfcorpus": 0.338, "fiqa": 0.356, "trec-covid": 0.738}),
    ("Contriever", "Contriever, unsupervised (HyDE T2)", "gao2023hyde",
     {"scifact": 0.649, "nfcorpus": None, "fiqa": 0.245, "trec-covid": 0.273}),
    ("HyDE-Contriever", "HyDE, text-davinci-003 + Contriever (HyDE T2)", "gao2023hyde",
     {"scifact": 0.691, "nfcorpus": None, "fiqa": 0.273, "trec-covid": 0.593}),
    ("RankGPT", "RankGPT, gpt-3.5-turbo on BM25 top-100 (RankGPT T1)", "sun2023rankgpt",
     {"scifact": 0.7043, "nfcorpus": 0.3562, "fiqa": None, "trec-covid": 0.7667}),
    (None, "BM25, Pyserini (RankGPT T1)", "sun2023rankgpt",
     {"scifact": 0.6789, "nfcorpus": 0.3075, "fiqa": None, "trec-covid": 0.5947}),
    (None, "RankGPT, gpt-4 on gpt-3.5 top-30 (RankGPT T1)", "sun2023rankgpt",
     {"scifact": 0.7495, "nfcorpus": 0.3847, "fiqa": None, "trec-covid": 0.8551}),
    (None, "monoT5 220M on BM25 top-100 (RankGPT T1)", "sun2023rankgpt",
     {"scifact": 0.7340, "nfcorpus": 0.3738, "fiqa": None, "trec-covid": 0.7834}),
    (None, "DPR, NQ-trained (BEIR T2)", "thakur2021beir",
     {"scifact": 0.318, "nfcorpus": 0.189, "fiqa": 0.112, "trec-covid": 0.332}),
    (None, "ColBERT v1 (BEIR T2)", "thakur2021beir",
     {"scifact": 0.671, "nfcorpus": 0.305, "fiqa": 0.317, "trec-covid": 0.677}),
]

# BEIR Table 9 Recall@100; TREC-COVID values are "capped" (denominator min(|rel|, 100)).
RECALL100 = [
    ("BM25-BEIR", "BM25 (BEIR T9)", "thakur2021beir",
     {"scifact": 0.908, "nfcorpus": 0.250, "fiqa": 0.539, "trec-covid": 0.498}),
    ("Dense", "bge-small-en-v1.5 (model card, MTEB; uncapped)", "bge2023card",
     {"scifact": 0.94167, "nfcorpus": 0.31135, "fiqa": 0.69552, "trec-covid": 0.13297}),
]

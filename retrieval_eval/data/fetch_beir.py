"""Download the BEIR test collections used in this spike (stdlib only) into data_beir/{name}/.

Source: BEIR (Thakur et al., 2021), official UKP mirror. Each zip holds corpus.jsonl,
queries.jsonl and qrels/{split}.tsv.
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
BEIR = ["scifact", "nfcorpus", "fiqa", "trec-covid"]
DEST = Path(__file__).resolve().parents[2] / "data_beir"


def main(names):
    """Download and unzip each named BEIR dataset into DEST, skipping ones already present."""
    DEST.mkdir(exist_ok=True)
    for n in names:
        if (DEST / n / "corpus.jsonl").exists():
            print(f"{n}: present")
            continue
        z = DEST / f"{n}.zip"
        print(f"{n}: downloading {URL.format(n)}", flush=True)
        urllib.request.urlretrieve(URL.format(n), z)
        with zipfile.ZipFile(z) as f:
            for m in f.namelist():  # refuse path traversal
                assert not m.startswith("/") and ".." not in Path(m).parts, m
            f.extractall(DEST)
        z.unlink()  # remove the zip once extracted
        print(f"{n}: extracted", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or BEIR)  # dataset names from the command line, default all four

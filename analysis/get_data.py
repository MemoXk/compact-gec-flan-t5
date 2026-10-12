"""Download the public benchmark data needed by analysis/verify.py into ./data (nothing is redistributed by this repository).

  * CoNLL-2014 shared-task test data (NUS)                      -> data/conll14st-test-data/
  * W&I+LOCNESS v2.1 (BEA-2019 shared task, Univ. of Cambridge)  -> data/wi+locness/
      Licence: non-commercial research and education only; read data/wi+locness/licence.wi.txt after download.
  * JFLEG test sources, references and the official GLEU script -> data/jfleg/
"""
import sys, tarfile, urllib.request
from pathlib import Path

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(parents=True, exist_ok=True)
def fetch(url, dest):
    if not dest.exists():
        print("downloading", url); urllib.request.urlretrieve(url, dest)
    return dest
def untar(tgz):
    with tarfile.open(tgz) as t:
        try: t.extractall(DATA, filter="data")
        except TypeError: t.extractall(DATA)
if not (DATA / "conll14st-test-data").exists():
    untar(fetch("https://www.comp.nus.edu.sg/~nlp/conll14st/conll14st-test-data.tar.gz", DATA / "conll14.tgz"))
if not (DATA / "wi+locness").exists():
    untar(fetch("https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz", DATA / "wi.tgz"))
    print("NOTE: W&I+LOCNESS is licensed for non-commercial research and education only (see data/wi+locness/licence.wi.txt).")
jf = DATA / "jfleg"; jf.mkdir(exist_ok=True)
for n in ["test/test.src"] + [f"test/test.ref{k}" for k in range(4)] + ["eval/gleu.py"]:
    fetch("https://raw.githubusercontent.com/keisks/jfleg/master/" + n, jf / Path(n).name)
print("data ready in", DATA)

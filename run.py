import argparse
import os
from pathlib import Path
from src.data import ROOT
os.environ.setdefault("HF_HOME",str(ROOT/".cache"/"huggingface"))
from src.experiment import evaluate, run

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["demo", "reproduce", "evaluate"])
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    out = a.output or ROOT / "results" / ("demo" if a.action == "demo" else "formal")
    if a.action == "evaluate":
        evaluate(out)
    else:
        run(out, a.action == "demo")

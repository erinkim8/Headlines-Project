from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def load_finbert_csv(filename="headlines_finbert.csv", subdir="raw"):
    """
    Load FinBERT-scored headlines from data/<subdir>/<filename>.
    Returns (df, path).
    """
    path = ROOT / "data" / subdir / filename
    df = pd.read_csv(path)
    return df, path

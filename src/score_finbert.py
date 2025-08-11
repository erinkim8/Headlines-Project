import argparse
import math
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

MODEL_NAME = "ProsusAI/finbert"  # Financial-sentiment-tuned BERT

def load_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        task="sentiment-analysis",
        return_all_scores=True,
        truncation=True,
        max_length=128,
        device=-1  # CPU; set to 0 if you have CUDA and want GPU
    )

def main(csv_path: str, text_col: str, out_path: str, batch_size: int = 32):
    df = pd.read_csv(csv_path)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found. Columns: {list(df.columns)}")

    texts = df[text_col].astype(str).fillna("").tolist()
    pipe = load_pipeline()

    # Run in batches to control memory
    results = []
    n = len(texts)
    num_batches = math.ceil(n / batch_size)
    for i in tqdm(range(num_batches), desc="Scoring with FinBERT"):
        chunk = texts[i*batch_size:(i+1)*batch_size]
        # Each item in 'out' is a list of dicts: [{'label': 'positive', 'score': ...}, ...]
        out = pipe(chunk)
        results.extend(out)

    # Convert scores to columns
    # FinBERT labels: 'positive', 'negative', 'neutral'
    labels = []
    negs, neus, poss, confidences = [], [], [], []
    for item in results:
        # item: list of dicts with all scores
        label_scores = {d["label"].lower(): float(d["score"]) for d in item}
        # some variants use caps; normalize keys
        neg = label_scores.get("negative", label_scores.get("NEGATIVE", 0.0))
        neu = label_scores.get("neutral", label_scores.get("NEUTRAL", 0.0))
        pos = label_scores.get("positive", label_scores.get("POSITIVE", 0.0))

        # top label & confidence
        pairs = [("negative", neg), ("neutral", neu), ("positive", pos)]
        top_label, top_score = max(pairs, key=lambda x: x[1])

        labels.append(top_label)
        negs.append(neg)
        neus.append(neu)
        poss.append(pos)
        confidences.append(top_score)

    df["finbert_label"] = labels
    df["finbert_neg"] = negs
    df["finbert_neu"] = neus
    df["finbert_pos"] = poss
    df["finbert_confidence"] = confidences

    # Save
    out_path = out_path or str(Path(csv_path).with_name(Path(csv_path).stem + "_finbert.csv"))
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to headlines.csv")
    ap.add_argument("--text-col", default="headline", help="Column containing the headline text")
    ap.add_argument("--out", default="", help="Output CSV path; default = <name>_finbert.csv next to input")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    main(args.csv, args.text_col, args.out, args.batch_size)

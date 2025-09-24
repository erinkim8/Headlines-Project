import pandas as pd

SENT_COLS = ["finbert_neg", "finbert_neu", "finbert_pos", "finbert_confidence"]

def parse_dates(df, col="Date"):
    """Parse dates robustly (day-first) and leave invalid as NaT."""
    df = df.copy()
    df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    return df

def validate_probs(df, cols=SENT_COLS):
    """Ensure probabilities are within [0,1] (NaNs allowed)."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            ok = df[c].between(0, 1) | df[c].isna()
            if not ok.all():
                bad = df.loc[~ok, [c]].head()
                raise ValueError(f"{c} has values outside [0,1]. Examples:\n{bad}")
    return df

def basic_info(df):
    """Quick summary dict you can print or save."""
    return {
        "rows": len(df),
        "cols": list(df.columns),
        "na_counts": df.isna().sum().to_dict()
    }

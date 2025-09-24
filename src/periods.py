import yaml

def tag_periods(df, date_col="Date", yaml_path="config/periods.yaml"):
    """
    Add 'period_type' (crisis/normal) and 'period_name' using config ranges.
    Rows with NaT in Date remain 'normal' by default unless you drop them first.
    """
    df = df.copy()
    df["period_type"] = "normal"
    df["period_name"] = "Normal"

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    for p in cfg["crisis_periods"]:
        mask = (df[date_col] >= p["start"]) & (df[date_col] <= p["end"])
        df.loc[mask, "period_type"] = "crisis"
        df.loc[mask, "period_name"] = p.get("name", "Crisis")
    return df

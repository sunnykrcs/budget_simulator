import pandas as pd

REQUIRED_COLUMNS = [
    "date",
    "spend",
    "campaign_name"
]

def validate(df, source):

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(
            f"{source} missing columns: {missing}"
        )

    if df.empty:
        raise ValueError(f"{source} contains no rows")

    return True
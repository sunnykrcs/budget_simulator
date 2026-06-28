import pandas as pd
from src.config import CAMPAIGN_TYPE_MAP

CHANNEL_MAP = {
    "google": "Google Ads",
    "meta": "Meta Ads",
    "microsoft": "Microsoft Ads"
}


# def prepare(df, channel):

#     df = df.copy()

#     df["channel"] = CHANNEL_MAP[channel]

#     if "revenue" not in df.columns:

#         if "conversion" in df.columns:
#             df["revenue"] = df["conversion"]

#         else:
#             df["revenue"] = 0

#     df["roas"] = df["revenue"] / df["spend"].replace(0, 1)

#     return df

def prepare(df, channel):

    df = df.copy()

    df["channel"] = CHANNEL_MAP[channel]

    # --------------------------------------------------
    # Ensure every dataset has campaign_type
    # --------------------------------------------------
    if "campaign_type" not in df.columns:
        df["campaign_type"] = f"{CHANNEL_MAP[channel]} Unknown"

    df["campaign_type"] = (
        df["campaign_type"]
        .astype(str)
        .str.strip()
        .str.replace("_", " ", regex=False)
    )



    df["campaign_type"] = (
        df["campaign_type"]
        .replace(CAMPAIGN_TYPE_MAP)
    )

    # --------------------------------------------------
    # Revenue
    # --------------------------------------------------
    if "revenue" not in df.columns:

        if "conversion" in df.columns:
            df["revenue"] = df["conversion"]

        else:
            df["revenue"] = 0

    # --------------------------------------------------
    # ROAS
    # --------------------------------------------------
    df["roas"] = df["revenue"] / df["spend"].replace(0, 1)

    return df


def merge(google, meta, microsoft):

    return pd.concat(
        [
            google,
            meta,
            microsoft
        ],
        ignore_index=True
    )
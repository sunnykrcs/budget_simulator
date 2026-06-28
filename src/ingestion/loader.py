import pandas as pd

from src.config import COLUMN_MAPPING


def load_csv(path, source):
    """
    Load and standardize a CSV.

    Parameters
    ----------
    path : str | Path
    source : str
        "google", "meta", or "microsoft"
    """

    df = pd.read_csv(path)

    df = df.rename(
        columns=COLUMN_MAPPING[source]
    )
    

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df



    
# import pandas as pd

# # def load_csv(path):
# #     """
# #     Load a CSV file with automatic date parsing.
# #     """
# #     df = pd.read_csv(path)

# #     if "date" in df.columns:
# #         df["date"] = pd.to_datetime(df["date"])

# #     elif "date_start" in df.columns:
# #         df["date"] = pd.to_datetime(df["date_start"])

# #     return df



# from src.config import COLUMN_MAPPING

# def load_csv(path):
#     df = pd.read_csv(path)

#     df = df.rename(columns=COLUMN_MAPPING)

#     if "date" in df.columns:
#         df["date"] = pd.to_datetime(df["date"])

#     return df


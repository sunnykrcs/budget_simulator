import pandas as pd


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate campaign-level data by date, channel,
    campaign type, and campaign name.
    """

    grouped = (
        df.groupby(
            [
                "date",
                "channel",
                "campaign_type",
                "campaign_name",
                "campaign_id"
            ],
            as_index=False,
        )
        .agg(
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
            clicks=("clicks", "sum"),
            impressions=("impressions", "sum"),
            conversions=("conversions", "sum"),
        )
    )

    grouped["roas"] = (
        grouped["revenue"]
        / grouped["spend"].replace(0, 1)
    )

    return grouped

# import pandas as pd

# def aggregate(df):

#     grouped = (

#         df.groupby(
#             [
#                 "date",
#                 "channel",
#                 "campaign_name"
#             ],
#             as_index=False
#         )

#         .agg(
#             spend=("spend", "sum"),
#             revenue=("revenue", "sum"),
#             clicks=("clicks", "sum"),
#             impressions=("impressions", "sum"),
#             conversions=("conversion", "sum")
#         )

#     )

#     grouped["roas"] = (
#         grouped["revenue"] /
#         grouped["spend"].replace(0, 1)
#     )

#     return grouped
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"

GOOGLE_FILE = DATA_DIR / "google_ads_campaign_stats.csv"
META_FILE = DATA_DIR / "meta_ads_campaign_stats.csv"
MS_FILE = DATA_DIR / "bing_campaign_stats.csv"


COLUMN_MAPPING = {
    "google": {
        "segments_date": "date",
        "campaign_name": "campaign_name",
        "campaign_advertising_channel_type": "campaign_type",
        "metrics_cost_micros": "spend",
        "metrics_conversions_value": "revenue",
        "metrics_clicks": "clicks",
        "metrics_impressions": "impressions",
        "metrics_conversions": "conversions",
        "campaign_id":"campaign_id"
    },

    "meta": {
        "date_start": "date",
        "campaign_name": "campaign_name",
        "spend": "spend",
        "conversion": "revenue",
        "clicks": "clicks",
        "impressions": "impressions",
        "campaign_id":"campaign_id"
    },

    "microsoft": {
        "TimePeriod": "date",
        "CampaignName": "campaign_name",
        "CampaignType": "campaign_type",
        "Spend": "spend",
        "Revenue": "revenue",
        "Clicks": "clicks",
        "Impressions": "impressions",
        "Conversions": "conversions",
        "CampaignId":"campaign_id"
    },
}

CAMPAIGN_TYPE_MAP = {
    "SEARCH": "Search",
    "Search": "Search",

    "SHOPPING": "Shopping",
    "Shopping": "Shopping",

    "DISPLAY": "Display",
    "Display": "Display",

    "VIDEO": "Video",
    "Video": "Video",

    "AUDIENCE": "Audience",
    "Audience": "Audience",

    "DEMAND GEN": "Demand Gen",
    "Demand_Gen": "Demand Gen",

    "PERFORMANCE MAX": "Performance Max",
    "Performance Max": "Performance Max",
    "Performance_Max": "Performance Max",
    "PerformanceMax": "Performance Max",
}

# =============================================================================
# Feature Pipeline Configuration
# =============================================================================

# Continuous numerical features that should be standardized before modeling.
STANDARD_SCALE_COLUMNS = [
    "spend",
    "revenue",
    "clicks",
    "impressions",
    "conversions",
    "roas",
    "ctr",
    "cpc",
    "cpm",
    "conversion_rate",
    "log_spend",
    "sqrt_spend",
    "spend_squared",
    "log_revenue",
    "revenue_growth",
    "revenue_share",
    "adstock_spend",
    "hill_spend",
    "logistic_spend",
]

# Prefixes for automatically generated engineered features
STANDARD_SCALE_PREFIXES = [
    "spend_lag_",
    "revenue_lag_",
    "spend_mean_",
    "spend_std_",
    "spend_min_",
    "spend_max_",
    "revenue_mean_",
    "revenue_std_",
    "revenue_min_",
    "revenue_max_",
]

OUTPUT_FILE = DATA_DIR / "master_dataset.csv"
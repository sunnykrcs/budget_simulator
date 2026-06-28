from src.config import *

from src.ingestion.loader import load_csv
from src.ingestion.validator import validate
from src.preprocessing.cleaner import clean,clean_google
from src.preprocessing.merger import prepare, merge
from src.preprocessing.aggregator import aggregate


def main():

    google = load_csv(GOOGLE_FILE, "google")
    meta = load_csv(META_FILE, "meta")
    microsoft = load_csv(MS_FILE, "microsoft")
    # google = load_csv(GOOGLE_FILE)
    # meta = load_csv(META_FILE)
    # microsoft = load_csv(MS_FILE)

    google = clean_google(google)

    validate(google, "Google")
    validate(meta, "Meta")
    validate(microsoft, "Microsoft")

    google = clean(google)
    meta = clean(meta)
    microsoft = clean(microsoft)

    google = prepare(google, "google")
    meta = prepare(meta, "meta")
    microsoft = prepare(microsoft, "microsoft")

    dataset = merge(
        google,
        meta,
        microsoft
    )

    dataset = aggregate(dataset)

    dataset.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(dataset.head())

    print("\nDataset shape:", dataset.shape)


if __name__ == "__main__":
    main()
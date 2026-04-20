"""Cloud Run Job entrypoint.

Downloads the latest Binance aggregated-trade Parquet file from a GCS source
bucket, optionally de-duplicates and sorts the data, then uploads the result
to a GCS destination bucket.

Required environment variables:
    GCS_SOURCE_BUCKET  - bucket that holds raw trade files
    GCS_SOURCE_PREFIX  - blob name prefix (e.g. aggregated_trades/BTCUSDT-aggTrades-)
    GCS_DEST_BUCKET    - bucket to write processed files into
    GCS_DEST_PREFIX    - prefix for output blobs (e.g. processed/BTCUSDT-aggTrades-)

Authentication:
    Set GOOGLE_APPLICATION_CREDENTIALS locally.
    On Cloud Run, attach a service account with roles/storage.objectAdmin.
"""

import logging
import os
import sys
from datetime import datetime, timezone

from config.googleCloud import (
    download_to_dataframe,
    get_latest_blob_path,
    upload_dataframe,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        log.error("Required environment variable %s is not set.", name)
        sys.exit(1)
    return value


def run():
    source_bucket = _require_env("GCS_SOURCE_BUCKET")
    source_prefix = _require_env("GCS_SOURCE_PREFIX")
    dest_bucket = _require_env("GCS_DEST_BUCKET")
    dest_prefix = _require_env("GCS_DEST_PREFIX")

    log.info("Looking for latest blob under gs://%s/%s", source_bucket, source_prefix)
    blob_name = get_latest_blob_path(source_bucket, source_prefix)
    if blob_name is None:
        log.error("No blobs found under prefix %s in bucket %s.", source_prefix, source_bucket)
        sys.exit(1)
    log.info("Latest blob: %s", blob_name)

    log.info("Downloading %s ...", blob_name)
    df = download_to_dataframe(source_bucket, blob_name)
    log.info("Downloaded %d rows.", len(df))

    # Lightweight processing: remove duplicates and sort by first column (timestamp)
    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        log.info("Dropped %d duplicate rows.", before - len(df))
    df = df.sort_values(by=df.columns[0]).reset_index(drop=True)

    # Build destination blob name: <prefix><original-filename-without-prefix>
    original_filename = blob_name.split("/")[-1]
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest_blob_name = f"{dest_prefix.rstrip('/')}/{run_ts}_{original_filename}"

    log.info("Uploading %d rows to gs://%s/%s ...", len(df), dest_bucket, dest_blob_name)
    upload_dataframe(dest_bucket, dest_blob_name, df)
    log.info("Job completed successfully.")


if __name__ == "__main__":
    run()

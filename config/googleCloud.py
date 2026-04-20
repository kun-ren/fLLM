import io
import logging

import pandas as pd
from google.cloud import storage


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Download a blob from GCS to a local file using ADC."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    logging.info(msg=f"FILE: {source_blob_name} has been downloaded to {destination_file_name}.")


def download_to_dataframe(bucket_name, blob_name) -> pd.DataFrame:
    """Download a Parquet blob from GCS and return it as a DataFrame."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    content = blob.download_as_bytes()
    return pd.read_parquet(io.BytesIO(content))


def get_latest_blob_path(bucket_name, prefix):
    """Return the lexicographically last blob name under the given prefix."""
    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    blob_list = sorted([b.name for b in blobs])
    if blob_list:
        return blob_list[-1]
    return None


def upload_blob(bucket_name, destination_blob_name, source_file_name):
    """Upload a local file to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    logging.info(msg=f"FILE: {source_file_name} uploaded to {destination_blob_name}.")


def upload_dataframe(bucket_name, blob_name, df: pd.DataFrame):
    """Upload a DataFrame as a Parquet blob to GCS without writing to disk."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    blob.upload_from_file(buffer, content_type="application/octet-stream")
    logging.info(msg=f"DataFrame uploaded to gs://{bucket_name}/{blob_name} ({len(df)} rows).")

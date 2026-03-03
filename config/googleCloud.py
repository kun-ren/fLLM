import pandas as pd
from google.cloud import storage
import io
import logging

key_path = "google_cloud_key.json"
storage_client = storage.Client.from_service_account_json(key_path)


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """download a file (blob) from GCS"""

    # get storage bucket
    bucket = storage_client.bucket(bucket_name)

    # get file object
    blob = bucket.blob(source_blob_name)

    # execute download
    blob.download_to_filename(destination_file_name)

    logging.info(f"FILE: {source_blob_name} has been downloaded to {destination_file_name}。")


def download_to_dataframe(bucket_name, blob_name) -> pd.DataFrame:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # convert file to byte stream
    content = blob.download_as_bytes()

    # wrap the stream with BytesIO and read with pandas
    df = pd.read_parquet(io.BytesIO(content))
    return df


def get_latest_blob_path(bucket_name, prefix):
    client = storage.Client()
    # list all files started with the prefix
    blobs = client.list_blobs(bucket_name, prefix=prefix)

    blob_list = sorted([b.name for b in blobs])

    if blob_list:
        return blob_list[-1]
    return None



latest_path = get_latest_blob_path("你的桶名", "aggregated_trades/BTCUSDT-aggTrades-")

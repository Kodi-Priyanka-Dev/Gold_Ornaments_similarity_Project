"""
Copy every blob from the local Azurite emulator (source of truth right now -
Jewellery_Data/ was deleted from local disk after backing up here) into a
real Azure Storage account container.

Source: AZURE_STORAGE_CONNECTION_STRING (Azurite, from .env)
Dest:   AZURE_STORAGE_DEST_ACCOUNT_URL + DefaultAzureCredential
        (uses your `az login` session, or a managed identity if run from
        inside Azure - no storage key needed for the destination)

Usage:
    az login   # if running from your own machine
    export AZURE_STORAGE_DEST_ACCOUNT_URL="https://<account>.blob.core.windows.net"
    python backend/scripts/migrate_to_real_azure.py
"""
import os
import sys
import time

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

SOURCE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER", "jewellery-data")
DEST_ACCOUNT_URL = os.getenv("AZURE_STORAGE_DEST_ACCOUNT_URL")
DEST_CONTAINER_NAME = os.getenv("AZURE_STORAGE_DEST_CONTAINER", CONTAINER_NAME)


def main():
    if not SOURCE_CONNECTION_STRING:
        print("AZURE_STORAGE_CONNECTION_STRING is not set (should point at Azurite). Aborting.")
        return
    if not DEST_ACCOUNT_URL:
        print("AZURE_STORAGE_DEST_ACCOUNT_URL is not set. "
              "Set it to https://<your-account>.blob.core.windows.net and re-run.")
        return

    source_container = BlobServiceClient.from_connection_string(
        SOURCE_CONNECTION_STRING
    ).get_container_client(CONTAINER_NAME)

    dest_service = BlobServiceClient(account_url=DEST_ACCOUNT_URL, credential=DefaultAzureCredential())
    dest_container = dest_service.get_container_client(DEST_CONTAINER_NAME)
    if not dest_container.exists():
        dest_container.create_container()
        print(f"Created destination container '{DEST_CONTAINER_NAME}'.")

    blobs = list(source_container.list_blobs())
    total = len(blobs)
    print(f"Found {total} blobs in Azurite container '{CONTAINER_NAME}'. "
          f"Copying to {DEST_ACCOUNT_URL} / '{DEST_CONTAINER_NAME}'...\n")

    started = time.time()
    copied = 0
    failed = 0
    for i, blob in enumerate(blobs, 1):
        try:
            data = source_container.download_blob(blob.name).readall()
            dest_container.upload_blob(name=blob.name, data=data, overwrite=True)
            copied += 1
        except Exception as e:
            print(f"FAILED: {blob.name}: {e}")
            failed += 1
        if i % 100 == 0 or i == total:
            elapsed = time.time() - started
            print(f"[{i}/{total}]  copied={copied} failed={failed}  ({elapsed:.1f}s)")

    print(f"\nDone in {time.time() - started:.1f}s. Copied {copied}/{total} ({failed} failed).")


if __name__ == "__main__":
    main()

"""Azure Blob Storage backing for jewelry images.

Local disk (Jewellery_Data/, backend/new_inventory/) stays the primary,
fast-path source for images that are already present. This module only
adds two things on top of that: (1) ingestion also uploads a copy to Blob
Storage, and (2) serving falls back to Blob Storage when a file is not
found locally (e.g. disk was cleared, or a container/host never had the
dataset baked in). Qdrant payloads and metadata.json are untouched - the
blob name is derived from the same relative path already stored as
`image_url`, so no data migration is required for this to work.

Auth (in priority order, so production never needs a shared key):
  1. AZURE_STORAGE_ACCOUNT_URL + Managed Identity (DefaultAzureCredential)
     - the recommended production setup on Azure (App Service/AKS/VM with
       a system- or user-assigned identity granted Storage Blob Data
       Contributor on the account). No secret in the environment at all.
  2. AZURE_STORAGE_CONNECTION_STRING - local dev against Azurite, or any
     environment where a managed identity isn't available.

If neither is configured, uploads are a no-op. Reads then fall back to the
container's public-read URL (AZURE_STORAGE_PUBLIC_URL; set it to "" to disable).

Blob layout is <Category>/<filename> (e.g. "Chains/unique mangalsutra.jpg") -
the old local Jewellery_Data/ folder level does not exist in the container.
"""
import os
import re
import threading
import urllib.parse
import urllib.request

BASE_DIR_FWD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace('\\', '/') + '/'
_KNOWN_PREFIXES = ["D:/Gold searching/", "E:/Projects/GitHub/GSS/", BASE_DIR_FWD]

_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER", "jewellery-data")
_ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL")  # e.g. https://<account>.blob.core.windows.net
_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

_DEFAULT_PUBLIC_URL = (f"{_ACCOUNT_URL.rstrip('/')}/{_CONTAINER_NAME}" if _ACCOUNT_URL
                       else "https://goldsimilarity.blob.core.windows.net/jewellery-data")
_PUBLIC_URL = os.getenv("AZURE_STORAGE_PUBLIC_URL", _DEFAULT_PUBLIC_URL).rstrip('/')
_PUBLIC_TIMEOUT_SECONDS = float(os.getenv("AZURE_STORAGE_PUBLIC_TIMEOUT_SECONDS", "10"))

_DATASET_DIR_RE = re.compile(r'(?:^|/)jewellery_data/', re.IGNORECASE)


def strip_dataset_prefix(path):
    """Drop everything up to and including a Jewellery_Data/ folder, whatever
    absolute prefix precedes it, leaving the container's <Category>/<filename>."""
    p = path.replace('\\', '/')
    match = _DATASET_DIR_RE.search(p)
    return p[match.end():] if match else p

_lock = threading.Lock()
_state = {"client": None, "checked": False, "enabled": bool(_ACCOUNT_URL or _CONNECTION_STRING)}


def is_enabled():
    return _state["enabled"]


def to_blob_key(local_path):
    """Map an absolute/relative local image path to a stable blob name,
    stripping the same host-specific prefixes _strip_image_url_prefix()
    already strips for display."""
    normalized = local_path.strip().strip("'\"").replace('\\', '/')
    for prefix in _KNOWN_PREFIXES:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    return strip_dataset_prefix(normalized).lstrip('/')


def _get_container_client():
    if not _state["enabled"]:
        return None
    if _state["client"] is not None:
        return _state["client"]
    with _lock:
        if _state["client"] is not None:
            return _state["client"]
        try:
            from azure.storage.blob import BlobServiceClient

            if _ACCOUNT_URL:
                from azure.identity import DefaultAzureCredential
                service_client = BlobServiceClient(account_url=_ACCOUNT_URL, credential=DefaultAzureCredential())
                print(f"[blob_storage] Using managed identity against {_ACCOUNT_URL}")
            else:
                service_client = BlobServiceClient.from_connection_string(_CONNECTION_STRING)
                print("[blob_storage] Using connection-string auth")

            container_client = service_client.get_container_client(_CONTAINER_NAME)
            if not _state["checked"]:
                if not container_client.exists():
                    container_client.create_container()
                _state["checked"] = True
            _state["client"] = container_client
        except Exception as e:
            print(f"[blob_storage] Disabled - failed to initialize: {e}")
            _state["enabled"] = False
            return None
    return _state["client"]


def upload_file(local_path, blob_key=None):
    """Upload a local file to Blob Storage. Never raises - ingestion must not
    fail just because Blob Storage is slow/unreachable; the file stays
    servable from local disk either way."""
    container = _get_container_client()
    if container is None:
        return False
    key = blob_key or to_blob_key(local_path)
    try:
        with open(local_path, "rb") as f:
            container.upload_blob(name=key, data=f, overwrite=True)
        return True
    except Exception as e:
        print(f"[blob_storage] Upload failed for {local_path!r} -> {key!r}: {e}")
        return False


def download_bytes(blob_key):
    """Fetch a blob's bytes, or None if missing/unavailable."""
    container = _get_container_client()
    if container is not None:
        try:
            return container.download_blob(blob_key).readall()
        except Exception:
            pass
    return _download_public(blob_key)


def _download_public(blob_key):
    if not _PUBLIC_URL:
        return None
    url = f"{_PUBLIC_URL}/{urllib.parse.quote(blob_key, safe='/')}"
    try:
        with urllib.request.urlopen(url, timeout=_PUBLIC_TIMEOUT_SECONDS) as resp:
            return resp.read()
    except Exception:
        return None

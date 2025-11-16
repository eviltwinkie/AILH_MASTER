# datagate_client.py
import os
import sys
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Optional

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import DATAGATE_USERNAME, DATAGATE_PASSWORD

HTTP_TIMEOUT = 60  # seconds

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)

async def safe_get(client: httpx.AsyncClient, url: str, headers: Optional[dict] = None, params: Optional[dict] = None) -> httpx.Response:
    return await client.get(url, headers=headers, params=params)

async def fetch_data(fetch_params: dict, datagate_url: str):
    params = {
        "Username": DATAGATE_USERNAME,
        "Password": DATAGATE_PASSWORD,
    }
    params.update(fetch_params)
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await safe_get(client, datagate_url, params=params)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "").lower()
        if content_type.startswith("text/") or \
           "application/json" in content_type or \
           "application/xml" in content_type or \
           "application/javascript" in content_type:
            return resp.text
        else:
            return resp.content

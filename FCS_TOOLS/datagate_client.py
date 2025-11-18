# datagate_client.py
import os
import sys
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Optional, Tuple

# Credential pairs: (username, password)
CREDENTIALS = [
    ("sbartal", "Sb749499houstonTX"),
    ("emartinez", "letmein2Umeow!!!"),
]

# Track current credential index
_current_cred_index = 0

def get_next_credentials() -> Tuple[str, str]:
    """Rotate to next credential pair and return (username, password)."""
    global _current_cred_index
    username, password = CREDENTIALS[_current_cred_index]
    _current_cred_index = (_current_cred_index + 1) % len(CREDENTIALS)
    return username, password

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)

async def safe_get(client: httpx.AsyncClient, url: str, headers: Optional[dict] = None, params: Optional[dict] = None) -> httpx.Response:
    resp = await client.get(url, headers=headers, params=params)
    resp.raise_for_status()
    return resp

async def fetch_data(fetch_params: dict, datagate_url: str):
    username, password = get_next_credentials()
    params = {
        "Username": username,
        "Password": password,
    }
    params.update(fetch_params)
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await safe_get(client, datagate_url, params=params)
        content_type = resp.headers.get("Content-Type", "").lower()
        if content_type.startswith("text/") or \
           "application/json" in content_type or \
           "application/xml" in content_type or \
           "application/javascript" in content_type:
            return resp.text
        else:
            return resp.content

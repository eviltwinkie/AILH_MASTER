# fileio.py

import os
import aiofiles
import json
import re
from config import LOGS_DIR, SENSOR_DATA, LAST_TIMESTAMP_FILE

async def get_last_timestamp():
    if not os.path.exists(LAST_TIMESTAMP_FILE):
        return None
    async with aiofiles.open(LAST_TIMESTAMP_FILE, mode='r') as f:
        data = await f.read()
        try:
            return json.loads(data).get("last_timestamp")
        except Exception:
            return None

async def save_last_timestamp(ts: str):
    async with aiofiles.open(LAST_TIMESTAMP_FILE, mode='w') as f:
        await f.write(json.dumps({"last_timestamp": ts}))

def sanitize_path_component(value: str) -> str:
    """Sanitize strings to be safe for filenames on all OSes."""
    if not isinstance(value, str):
        value = str(value)
    return re.sub(r'[<>:"/\\|?*\s]+', "_", value)

def check_output_files_exist(file_path):
    if os.path.exists(file_path):
        #print(f"[!] Skipping: output file already exists → {file_path}")
        return True
    return False

def create_file_path(sensor_dir, filename_base, requested_format=None, recording_time=None, recording_gain=None):
    if requested_format in ("json", "json_flot"):
        suffix = "flot.json" if requested_format == "json_flot" else "json"
        file_path = os.path.join(sensor_dir, f"{filename_base}.{suffix}")    
    elif requested_format == "csv":
        file_path = os.path.join(sensor_dir, f"{filename_base}.csv")
    elif requested_format == "wav":
        if recording_time is None or recording_gain is None:
            raise ValueError("Missing recording_time or recording_gain for WAV output")
        sanitized_time = sanitize_path_component(recording_time)
        sanitized_gain = sanitize_path_component(recording_gain)
        file_path = os.path.join(sensor_dir, f"{filename_base}~{sanitized_time}~{sanitized_gain}.wav")    
    else:
        file_path = os.path.join(sensor_dir, f"{filename_base}")
    return file_path
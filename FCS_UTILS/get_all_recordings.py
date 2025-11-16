
import os
import sys
import json
import asyncio
import argparse
from argparse import Namespace
from fcs_recordings import run as get_all_recordings
from fcs_get_recordings import run as run_get_recordings

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import SENSOR_DATA, LOGS_DIR

def load_json_file(file_path):
    """
    Load a JSON file and return its content as a dictionary.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Parsed JSON content as a Python dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[✗] File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            print(f"[✓] Successfully loaded JSON from: {file_path}")
            return data
        except json.JSONDecodeError as e:
            print(f"[✗] Failed to decode JSON: {e}")
            raise

def process_data(data):
    """
    Process the dictionary data as needed.

    Args:
        data (dict): Dictionary loaded from JSON.

    Example: Print all key-value pairs.
    """
    print("[ℹ] Processing data...")
    for key, value in data.items(): print(f"  {key}: {value}")


async def download_all_recordings(data):

    recording = data.get("recordings", {}).get("recording", [])
    if isinstance(recording, dict): recording = [recording]
    print(f"[✓] Found {len(recording)} recordings to process.")

    for object in recording:
        recording_id = object.get("id", -1)
        recording_gain = object.get("gain", -1)
        recording_time = object.get("rst", -1) 
        recording_siteid = object.get("siteid", -1)
        recording_siteidtext = object.get("siteidtext", -1)

        parts = recording_siteidtext.split()
        sensor_name = parts[0] if len(parts) > 0 else -1
        sensor_site_id = parts[1] if len(parts) > 1 else -1
        sensor_site_name = parts[2] if len(parts) > 2 else -1
        sensor_station = parts[3] if len(parts) > 3 else -1

        # WAV fetching
        print(f"[✓] Fetching WAV for sensor ID {recording_siteid} and recording ID {recording_id}...")
        getrec_args = Namespace(sensorId=recording_siteid, sensorName=sensor_name, siteId=sensor_site_id, siteName=sensor_site_name, siteStation=sensor_station, recordingId=recording_id, recordingGain=recording_gain, recordingTime=recording_time, reqFormat="wav")
        try:
            wav_bytes = await run_get_recordings(getrec_args)
            if isinstance(wav_bytes, bytes):
                print(f"[✓] WAV bytes received: {len(wav_bytes)} bytes")
            else:
                print(f"[!] run_get_recordings returned non-bytes: {type(wav_bytes)}")
        except Exception as e:
            print(f"[!] Exception while fetching WAV: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download FCS Recordings")
    parser.add_argument("--siteId", type=str, required=True, help="Download recordings with this siteId.")
    args = parser.parse_args()
    try:
        get_args = Namespace(
            lastId=0,
            siteId=args.siteId
        )
        asyncio.run(get_all_recordings(get_args))

        data = load_json_file(f"{LOGS_DIR}/fcs_recordings_{args.siteId}.json")
        asyncio.run(download_all_recordings(data))
    except Exception as e:
        print(f"[✗] Error: {e}")

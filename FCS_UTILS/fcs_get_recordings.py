# fcs_get_recordings.py

import os
import sys
import json
import csv
import re

from datagate_client import fetch_data
from UTILITIES.fileio import sanitize_path_component, create_file_path, check_output_files_exist

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import LOGS_DIR, SENSOR_DATA


GETRECORDINGS_URL = "https://api.omnicoll.net/datagate/api/getrecordingsapi.ashx"

def add_arguments(parser):
    parser.add_argument("--recordingId", type=str, required=True, help="Recording ID")
    parser.add_argument("--reqFormat", type=str, choices=["wav", "csv", "json", "json_flot"], required=False, default="wav", help="Requested Format: wav, csv, json, json_flot")
    parser.add_argument("--sensorId", type=str, required=False, default="000000", help="Sensor ID")
    parser.add_argument("--sensorName", type=str, required=False, default="DEV", help="Sensor Name")
    parser.add_argument("--siteId", type=str, required=False, default="XXX-000", help="Site ID")
    parser.add_argument("--siteName", type=str, required=False, default="YY-000", help="Site Name")
    parser.add_argument("--siteStation", type=str, required=False, default="0_000.000", help="Station Label")
    parser.add_argument("--recordingGain", type=str, required=False, default="000", help="Recording gain")
    parser.add_argument("--recordingTime", type=str, required=False, default="YYYY-MM-DD_HH_MM_SS.M", help="Recording time")

def save_wav(data, file_path):
    if not isinstance(data, (bytes, bytearray)): raise ValueError("Expected WAV binary content as bytes")
    with open(file_path, "wb") as f: f.write(data)
    print(f"[✓] WAV binary saved to: {file_path}")

async def run(args):
    #print(f"[✓] Running fcs_get_recordings with args: {args}")
    data = None
    recording_id     = args.recordingId
    requested_format = args.reqFormat
    sensor_id        = args.sensorId
    sensor_name      = args.sensorName
    sensor_site_id   = args.siteId
    sensor_site_name = args.siteName
    sensor_station   = args.siteStation
    recording_gain   = args.recordingGain
    recording_time   = args.recordingTime
    params = {
        "software": "Amsys",
        "id": recording_id,
        "format": requested_format,
    }

    # Organize directories by sanitized site/sensor structure
    site_dir = os.path.join(SENSOR_DATA, f"{sanitize_path_component(sensor_site_id)}_{sanitize_path_component(sensor_site_name)}_{sanitize_path_component(sensor_station)}")
    sensor_dir = os.path.join(site_dir, f"{sanitize_path_component(sensor_name)}_{sanitize_path_component(sensor_id)}")
    os.makedirs(site_dir, exist_ok=True)
    os.makedirs(sensor_dir, exist_ok=True)
    filename_base = f"{sensor_id}~{recording_id}"
    file_path = create_file_path(sensor_dir, filename_base, requested_format, recording_time, recording_gain)   

    if requested_format == "wav" and check_output_files_exist(file_path):
        #print("[!] File already exists. Skipping processing.")
        return 
    else:
        #print("[✓] Proceeding with API request...")  
        data = await fetch_data(params, GETRECORDINGS_URL)
        
        # JSON Output
        if requested_format in ("json", "json_flot"):
            # Parse data if it's a string
            if isinstance(data, str):
                try:
                    parsed = json.loads(data)
                except json.JSONDecodeError as e:
                    raise ValueError(f"[✗] Failed to parse JSON string: {e}")
            elif isinstance(data, (dict, list)):
                parsed = data
            else:
                raise TypeError("[✗] Unsupported data type for JSON export")
            # Save to disk
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2)
            print(f"[✓] JSON saved to: {file_path}")
            return parsed

        # CSV Output
        elif requested_format == "csv":
            # If 'data' is a string, split it into lines for CSV parsing
            if isinstance(data, str):
                lines = data.strip().splitlines()
                reader = csv.reader(lines)
                with open(file_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(reader)
                print(f"[✓] CSV saved to: {file_path}")
            # If 'data' is already a list of lists or rows
            elif isinstance(data, list):
                # Ensure each item is an iterable of columns (e.g., list or tuple)
                processed_rows = []
                for row in data:
                    if isinstance(row, (bytes, bytearray)):
                        # Decode and split by comma (assuming CSV format)
                        decoded = row.decode("utf-8", errors="replace")
                        processed_rows.append(decoded.strip().split(","))
                    elif isinstance(row, str):
                        processed_rows.append(row.strip().split(","))
                    elif isinstance(row, (list, tuple)):
                        processed_rows.append(row)
                    else:
                        processed_rows.append([str(row)])
                with open(file_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(processed_rows)
                print(f"[✓] CSV saved to: {file_path}")
            else:
                raise ValueError("[✗] Unsupported data type for CSV export")
            return data
        
        # WAV Output
        elif requested_format == "wav":
            if not isinstance(data, (bytes, bytearray)): raise ValueError("Expected WAV binary content as bytes")
            with open(file_path, "wb") as f: f.write(data)
            print(f"[✓] WAV binary saved to: {file_path}")
            return data

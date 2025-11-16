# fcs_loggers.py

import os
import sys
import json
import base64
import xmltodict

from datagate_client import fetch_data
from argparse import Namespace
from datetime import datetime, timedelta
from UTILITIES.fileio import get_last_timestamp, save_last_timestamp
from fcs_recordings import run as run_recordings
from fcs_get_recordings import run as run_get_recordings

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import LOGS_DIR

LOGGER_URL = "https://api.omnicoll.net/datagate/api/loggerapi.ashx"

def add_arguments(parser):
    """Reserved for future argparse extension."""
    pass

def filter_loggers(parsed_dict):
    loggers = parsed_dict.get("loggers", {}).get("logger", [])
    if isinstance(loggers, dict):
        loggers = [loggers]
    parsed_dict["loggers"]["logger"] = [
        l for l in loggers if l.get("loggerid") not in (None, "", "null")
    ]
    return parsed_dict

def parse_and_save_logger_xml(xml_data, last_timestamp, out_filename):

    try:
        parsed_dict = xmltodict.parse(xml_data)
    except Exception as e:
        print(f"[!] XML parsing failed: {e}")
        return {}, None

    if "loggers" not in parsed_dict:
        print("[!] 'loggers' key missing in parsed XML response. Dumping raw XML for inspection:")
        error_dump_path = os.path.join(LOGS_DIR, "fcs_loggers_raw_response.xml")
        with open(error_dump_path, "w", encoding="utf-8") as f:
            f.write(xml_data)
        print(f"[!] Raw XML response saved to: {error_dump_path}")
        return {}, None

    parsed_dict = filter_loggers(parsed_dict)

    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(out_filename, "w", encoding="utf-8") as json_file:
        json.dump(parsed_dict, json_file, indent=2)

    return parsed_dict, None  # Optional: could extract a max timestamp here

async def extract_sensor_data(parsed_dict):
    sensor_data_list = []

    loggers = parsed_dict.get("loggers", {}).get("logger", [])
    if isinstance(loggers, dict):
        loggers = [loggers]

    sensor_type_map = {
        "FW-155-001U": "MAG1",
        "FW-138-007U": "MAG2",
        "FW-138-006U": "HYD2"
    }

    for logger in loggers:
        sensor_id = logger.get("loggerid", -1)
        site_notes = logger.get("SiteNotes", -1)
        serial_number = logger.get("serialNumber", -1)
        logger_type = logger.get("type", "")
        sensor_type = sensor_type_map.get(logger_type, -1)

        site_id_raw = logger.get("siteId", "")
        parts = site_id_raw.split()
        sensor_name     = parts[0] if len(parts) > 0 else -1
        sensor_site_id  = parts[1] if len(parts) > 1 else -1
        sensor_site_name = parts[2] if len(parts) > 2 else -1
        sensor_station   = parts[3] if len(parts) > 3 else -1

        # Initialize measurement values
        leak_val = noise_val = spread_val = pressure_val = -1
        leak_unit = noise_unit = spread_unit = pressure_unit = -1
        meter_read_date = audio_read_date = None

        channels = logger.get("channels", {}).get("channel", [])
        if isinstance(channels, dict):
            channels = [channels]

        for ch in channels:
            try:
                measurement = ch.get("measurement") or {}

                # Safely extract fields with defaults
                measurement_name = (measurement.get("name") or "").strip()

                last_value = ch.get("lastValue")
                last_value = last_value if last_value not in (None, "", "null", "NULL", "NaN") else -1

                raw_date = ch.get("lastValueDate")
                if str(raw_date).strip() not in ("", "-1", "null", "None", "NULL"):
                    meter_read_date = raw_date

                unit = measurement.get("units", {}).get("symbols")
                unit = unit if unit not in (None, "", "null", "NULL") else "-"

                #print(f"[✓] Processing logger {sensor_id} - Measurement: {measurement_name}, Last Value: {last_value}, meter_read_date: {meter_read_date}, Unit: {unit}")

                try:
                    if isinstance(meter_read_date, str):
                        dt = datetime.strptime(meter_read_date, "%Y-%m-%d %H:%M:%S.%f")
                        audio_read_date = (dt - timedelta(minutes=45)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    else:
                        audio_read_date = meter_read_date
                except Exception as e:
                    print(f"[!] Failed to parse meter_read_date for logger {sensor_id}: {e}")
                    audio_read_date = meter_read_date

                if measurement_name == "Leak":
                    leak_val, leak_unit = last_value, unit
                elif measurement_name == "Noise":
                    noise_val, noise_unit = last_value, unit
                elif measurement_name == "Spread":
                    spread_val, spread_unit = last_value, unit
                elif measurement_name == "Pressure":
                    pressure_val, pressure_unit = last_value, unit
            except Exception:
                continue

        # Fetch associated recordings
        recording_time = -1
        recording_gain = -1
        recordings_data = {}
        wav_base64_data = None  # Initialize to avoid unbound error

        if isinstance(sensor_id, str) and sensor_id.isdigit():
            sensor_id = int(sensor_id)

        if isinstance(sensor_id, int) and sensor_id > 0:
            recording_args = Namespace(
                #lastId=0,
                siteId=sensor_id,
                StartDate=audio_read_date
            )
            recordings_data = await run_recordings(recording_args) or {}

        recording = recordings_data.get("recordings", {}).get("recording", {})

        print(f"[✓] Processing recordings for sensor ID {sensor_id} at {audio_read_date}")
        #print(recording)   

        if isinstance(recording, list):
            recording = recording[0]  # just take the first one

        recording_time = recording.get("rst", -1) if isinstance(recording, dict) else -1
        recording_gain = recording.get("gain", -1) if isinstance(recording, dict) else -1
        recording_id = -1

        if isinstance(recording, dict):
            raw_id = recording.get("id")
            try:
                if raw_id is not None:
                    recording_id = int(raw_id)
                else:
                    raise ValueError("recording id is None")
            except (ValueError, TypeError):
                #print(f"[!] Invalid recording ID: {raw_id}")
                recording_id = -1
                
        #print(f"Sensor ID {sensor_id} at {audio_read_date} recording ID {recording_id}")                

        # WAV fetching
        #print(f"[✓] Fetching WAV for recording ID {recording_id} from get_recordings")
        if isinstance(recording_id, int) and recording_id > 0:
            #print(f"[✓] Fetching WAV for sensor ID {sensor_id} at {audio_read_date} recording ID {recording_id} from get_recordings")
            getrec_args = Namespace(sensorId=sensor_id, sensorName=sensor_name, siteId=sensor_site_id, siteName=sensor_site_name, siteStation=sensor_station, recordingId=recording_id, recordingGain=recording_gain, recordingTime=recording_time, reqFormat="wav")
            try:
                wav_bytes = await run_get_recordings(getrec_args)
                if isinstance(wav_bytes, bytes):
                    wav_base64_data = base64.b64encode(wav_bytes).decode("utf-8")
                    #print(f"[✓] WAV bytes received: {len(wav_bytes)} bytes")
                else:
                   print(f"[!] run_get_recordings returned non-bytes: {type(wav_bytes)}")
            except Exception as e:
                print(f"[!] Exception while fetching WAV: {e}")
                wav_base64_data = -1

        sensor_data = {
            "sensorId": sensor_id,
            "sensorSN": serial_number,
            "sensorType": sensor_type,
            "sensorName": sensor_name,
            "siteId": sensor_site_id,
            "siteName": sensor_site_name,
            "siteStation": sensor_station,
            "siteNeighbor": site_notes,
            "batteryCondition": logger.get("batteryCondition", -1),
            "signalStrength": logger.get("signalStrength", -1),
            "latitude": logger.get("latitude", -1),
            "longitude": logger.get("longitude", -1),
            "meterReadDate": meter_read_date,
            "leakVal": leak_val,
            "leakUnit": leak_unit,
            "noiseVal": noise_val,
            "noiseUnit": noise_unit,
            "spreadVal": spread_val,
            "spreadUnit": spread_unit,
            "pressureVal": pressure_val,
            "pressureUnit": pressure_unit,
            "recordingId": recording_id,
            "recordingTime": recording_time,
            "recordingGain": recording_gain,
            "recordingWavData": wav_base64_data,  # Base64-encoded WAV binary
        }

        sensor_data_list.append(sensor_data)

    return sensor_data_list

# ------------------------
# Entry Function
# ------------------------

async def run(args):
    params = {
        "ShowAssociations": "false",
        "ShowSubAccounts": "false",
        "SummaryOnly": "false"
    }

    xml_data = await fetch_data(params, LOGGER_URL)
    last_ts = await get_last_timestamp()

    out_filename = os.path.join(LOGS_DIR, "fcs_loggers_data.json")
    sensor_output_file = os.path.join(LOGS_DIR, "sensorPayload.json")

    parsed_data, max_ts = parse_and_save_logger_xml(xml_data, last_ts, out_filename)

    if max_ts:
        await save_last_timestamp(max_ts)

    sensor_data = await extract_sensor_data(parsed_data)

    with open(sensor_output_file, "w", encoding="utf-8") as f:
        json.dump(sensor_data, f, indent=2)

    print(f"[✓] Logger data written to: {out_filename}")
    print(f"[✓] Sensor payload written to: {sensor_output_file}")

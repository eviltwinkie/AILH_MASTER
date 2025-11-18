import os
import re
import json
import base64
import xmltodict
import asyncio
from datetime import datetime, timedelta, UTC
from datagate_client import fetch_data

LOGS_DIR = "/DEVELOPMENT/ROOT_AILH/DATA_STORE/PROC_LOGS/FCS"
DATA_SENSORS = "/DEVELOPMENT/ROOT_AILH/DATA_SENSORS"
LOGGER_URL = "https://api.omnicoll.net/datagate/api/loggerapi.ashx"
RECORDINGS_URL = "https://api.omnicoll.net/datagate/api/recordingsapi.ashx"
GETRECORDINGS_URL = "https://api.omnicoll.net/datagate/api/getrecordingsapi.ashx"

def sanitize_path_component(value: str) -> str:
    """Sanitize strings to be safe for filenames on all OSes."""
    if not isinstance(value, str):
        value = str(value)
    return re.sub(r'[<>:"/\\|?*\s]+', "_", value)

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

async def get_accounts(parsed_dict):
    accounts_array = parsed_dict.get("loggers", {}).get("summary", {}).get("SubAccounts", {}).get("SubAccount", [])
    with open(os.path.join(LOGS_DIR, "fcs_accounts.json"), "w", encoding="utf-8") as json_file:
        json.dump(accounts_array, json_file, indent=2)
    #print(f"[✓] Total subaccounts found: {len(subaccounts)}")    
    #print(f"[✓] SubAccounts: {subaccounts}")    
    return accounts_array

async def get_loggers(parsed_dict):
    loggers_array = parsed_dict.get("loggers", {}).get("logger", [])
    with open(os.path.join(LOGS_DIR, "fcs_loggers.json"), "w", encoding="utf-8") as json_file:
        json.dump(loggers_array, json_file, indent=2)
    #print(f"[✓] Total loggers found: {len(loggers_array)}")    
    #print(f"[✓] Loggers: {loggers_array}")    
    return loggers_array

async def get_logger_recordings_list(loggers_array):

    for logger in loggers_array:
        logger_id = int(logger.get("id", -1))
        logger_owner = int(logger.get("owner", -1))
        #print(f"[✓] Fetching recordings for {logger_id} ({logger_owner})")
        json_path = os.path.join(LOGS_DIR, f"fcs_logger_{logger_id}_recordings.json")
        # -------------------------------------------------
        # LOAD EXISTING RECORDINGS (IF FILE ALREADY EXISTS)
        # -------------------------------------------------
        existing_recordings = []
        seen_ids = set()
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                raw = data.get("recordings", {}).get("recording", [])
                if isinstance(raw, dict):
                    raw = [raw]
                elif not isinstance(raw, list):
                    raw = []
                existing_recordings = raw
                for rec in existing_recordings:
                    try:
                        rid = int(rec.get("id", -1))
                        if rid >= 0:
                            seen_ids.add(rid)
                    except Exception:
                        continue
            except Exception as e:
                print(f"[!] Warning: Failed to parse existing JSON for logger {logger_id}: {e}")
        lastId = max(seen_ids)-1 if seen_ids else 0
        #print(f"[✓] Starting last recording ID for logger {logger_id} is {lastId}")
        # -------------------------------------------------
        # DATE RANGE PARAMETERS
        # -------------------------------------------------
        today = datetime.now(UTC).date()
        one_year_ago = today - timedelta(days=365)
        EndDate = today.strftime("%Y-%m-%d")
        StartDate = one_year_ago.strftime("%Y-%m-%d")
        #print(f"[✓] Date range: {StartDate} → {EndDate}")
        # --------------------------------
        # KEEP FETCHING UNTIL NO NEW DATA
        # --------------------------------
        while True:
            params = {
                "lastId": int(lastId),
                "siteId": int(logger_id),
                "StartDate": StartDate,
                "EndDate": EndDate
            }
            xml_data = await fetch_data(params, RECORDINGS_URL)
            parsed_dict = xmltodict.parse(xml_data)
            new_recs = (parsed_dict.get("recordings", {}) or {}).get("recording", []) # type: ignore
            if isinstance(new_recs, dict):
                new_recs = [new_recs]
            elif not isinstance(new_recs, list):
                new_recs = []
            if not new_recs:
                #print(f"[✓] No recordings returned from API for logger {logger_id} (lastId={lastId}).")
                break
            truly_new = []
            for rec in new_recs:
                try:
                    rid = int(rec.get("id", -1))
                except Exception:
                    rid = -1
                if rid >= 0 and rid not in seen_ids:
                    seen_ids.add(rid)
                    truly_new.append(rec)
            if not truly_new:
                #print(f"[✓] No NEW recordings to append for logger {logger_id}. Stopping.")
                break
            #print(f"[+] Appending {len(truly_new)} new recordings for logger {logger_id}")
            existing_recordings.extend(truly_new)
            existing_recordings.sort(
                key=lambda x: int(x.get("id", 0)) if str(x.get("id", "")).isdigit() else 0
            )
            lastId = max(seen_ids)
            #print(f"[✓] Updated lastId = {lastId}")
            output = {
                "recordings": {
                    "recording": existing_recordings
                }
            }
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(output, jf, indent=2)
            print(f"[✓] Saved {len(existing_recordings)} total recordings for logger {logger_id}")

        print(f"[✓] Finished fetching all available recordings for logger {logger_id}")

async def get_logger_recordings_data(loggers_array):

    def parse_siteidtext(siteidtext: str):
        # parts = siteidtext.strip().split()
        # logger_name = parts[0]
        # site_id = parts[1] if parts[1] not in ("null", "0", 0, None, "") else "OFF"
        # site_name = parts[2] if parts[2] not in ("null", "0", 0, None, "") else "OFF"
        # site_station = parts[3] if parts[3] not in ("null", "0", 0, None, "") else "OFF"

        if not isinstance(siteidtext, str):
            parts = []
        else:
            parts = siteidtext.strip().split()

        # Ensure we have exactly 4 positions (pad or truncate)
        if len(parts) < 4:
            parts = parts + ["OFF"] * (4 - len(parts))
        else:
            parts = parts[:4]

        # Normalize values: null-like → "OFF"
        normalized_parts = []
        for v in parts:
            if v is None:
                normalized_parts.append("OFF")
                continue
            v_str = str(v).strip()
            if v_str.lower() in ("", "null", "none", "0"):
                normalized_parts.append("OFF")
            else:
                normalized_parts.append(v_str)

        return normalized_parts[0], normalized_parts[1], normalized_parts[2], normalized_parts[3]
#        logger_name, site_id, site_name, site_station = normalized_parts
#        return logger_name, site_id, site_name, site_station


    for logger in loggers_array:
        logger_id = int(logger.get("id", -1))
        logger_owner = int(logger.get("owner", -1))
        
        print(f"[✓] Fetching recordings for {logger_id} ({logger_owner})")

        json_path = os.path.join(LOGS_DIR, f"fcs_logger_{logger_id}_recordings.json")

        print(f"[✓] Loading recordings from {json_path}")
        if not os.path.exists(json_path):
            print(f"[!] Recordings file does not exist for logger {logger_id}, skipping.")
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as jf:
                data = json.load(jf)
            recordings = data.get("recordings", {}).get("recording", [])
            if isinstance(recordings, dict):
                recordings = [recordings]
            elif not isinstance(recordings, list):
                recordings = []
        except Exception as e:
            print(f"[!] Failed to load recordings JSON for logger {logger_id}: {e}")
            continue

        print(f"[✓] Total recordings to process for logger {logger_id}: {len(recordings)}\n")
        #print(recordings) 

        for recording in recordings:
            recording_id = recording.get("id", 0)
            recording_rst = recording.get("rst", "0000-00-00 00:00:00")
            recording_siteidtext = recording.get("siteidtext", f"{recording_id} OFF OFF 0_0.0")
            recording_recordingType = recording.get("recordingType", 0)
            recording_gain = recording.get("gain", None)         
            #print(f"Processing logger {logger_id} at {recording_rst} recording ID {recording_id} siteidtext {recording_siteidtext} recordingType {recording_recordingType} gain {recording_gain}")

            if recording_recordingType != "3" or recording_siteidtext is None and recording_gain is None:
                #print(f"[!] Skipping non-audio recording ID {recording_id} of type {recording_recordingType}")      
                continue          
            else:
                # WAV fetching
                print(f"[✓] Fetching WAV for logger ID {logger_id} - recording ID {recording_id}")
                logger_name, site_id, site_name, site_station = parse_siteidtext(recording_siteidtext)                                   
                logger_params = {
                    "loggerId": logger_id,
                    "loggerName": logger_name,                        
                    "siteId": site_id,
                    "siteName": site_name,
                    "siteStation": site_station,
                    "recordingId": recording_id,
                    "recordingGain": recording_gain,
                    "recordingTime": recording_rst,
                    "reqFormat": "wav"
                }
                await run_get_recordings(logger_params)


async def run_get_recordings(logger_params):
    
    loggerId = logger_params.get("loggerId")
    logger_name = logger_params.get("loggerName")
    siteId = logger_params.get("siteId")
    siteName = logger_params.get("siteName")
    siteStation = logger_params.get("siteStation")
    recordingId = logger_params.get("recordingId")
    recording_rst = logger_params.get("recordingTime")
    recording_gain = logger_params.get("recordingGain")

    site_dir = os.path.join(DATA_SENSORS, f"{sanitize_path_component(siteId)}_{sanitize_path_component(siteName)}_{sanitize_path_component(siteStation)}")
    sensor_dir = os.path.join(site_dir, f"{sanitize_path_component(logger_name)}_{sanitize_path_component(loggerId)}")
    os.makedirs(site_dir, exist_ok=True)
    os.makedirs(sensor_dir, exist_ok=True)
    filename_base = f"{loggerId}~{recordingId}"
    file_path = create_file_path(sensor_dir, filename_base, "wav", recording_rst, recording_gain)   

    if os.path.exists(file_path):
        print("[!] File already exists. Skipping processing.")
        return True
    else:
        print("[✓] Proceeding with API request...")
        params = {
            "id": recordingId
        }   
        data = await fetch_data(params, GETRECORDINGS_URL)       
        if not isinstance(data, (bytes, bytearray)): raise ValueError("Expected WAV binary content as bytes")
        with open(file_path, "wb") as f: f.write(data)
        print(f"[✓] WAV binary saved to: {file_path}")
        return data

# ------------------------
# Entry Function
# ------------------------

async def main():
    
    params = {
        "ShowAssociations": "true",
        "ShowNestedLevels": "true",
        "ShowSubAccounts": "true",
        "SummaryOnly": "false"
    }
    xml_data = await fetch_data(params, LOGGER_URL)
    parsed_dict = xmltodict.parse(xml_data)
    with open(os.path.join(LOGS_DIR, "fcs_api_logger.json"), "w", encoding="utf-8") as json_file:
        json.dump(parsed_dict, json_file, indent=2)
    await get_accounts(parsed_dict)
    loggers_array = await get_loggers(parsed_dict)
    await get_logger_recordings_data(loggers_array)

if __name__ == "__main__":
    asyncio.run(main())

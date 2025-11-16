# fcs_recordings.py
import os
import sys 
import json
import xmltodict  # type: ignore
from datagate_client import fetch_data

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import SENSOR_DATA, LOGS_DIR

RECORDINGS_URL = "https://api.omnicoll.net/datagate/api/recordingsapi.ashx"

def add_arguments(parser):
    parser.add_argument(
        "--lastId",
        type=int,
        required=True,
        help="Last ID seen or '0' for all recordings"
    )
    parser.add_argument(
        "--siteId",
        type=int,
        required=False,
        help="Site ID to filter recordings"
    )
    parser.add_argument(
        "--StartDate",
        type=str,
        required=False,
        help="Start date to filter recordings"
    )
    parser.add_argument(
        "--EndDate",
        type=str,
        required=False,
        help="End date to filter recordings"
    )

async def run(args):
    params = {
        "lastId": getattr(args, "lastId", 0),
        "siteId": getattr(args, "siteId", None),
        "StartDate": getattr(args, "StartDate", None),
        "EndDate": getattr(args, "EndDate", None),
        "type": "3"  # Assuming type 3 is a fixed category of interest
    }

    try:
        data = await fetch_data(params, RECORDINGS_URL)
    except Exception as e:
        print(f"Error fetching data from API: {e}")
        return

    filename_base = f"fcs_recordings_{args.siteId}"
    out_json = os.path.join(LOGS_DIR, f"{filename_base}.json")
    out_err = os.path.join(LOGS_DIR, f"{filename_base}_err.xml")

    try:
        parsed = xmltodict.parse(data)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2)
        print(f"Recordings XML parsed and saved as JSON to {out_json}")
        return parsed
    except Exception as e:
        with open(out_err, "w", encoding="utf-8") as f:
            if isinstance(data, bytes):
                f.write(data.decode("utf-8", errors="replace"))
            else:
                if isinstance(data, bytearray):
                    f.write(data.decode("utf-8", errors="replace"))
                elif isinstance(data, memoryview):
                    f.write(data.tobytes().decode("utf-8", errors="replace"))
                else:
                    f.write(str(data))
        print(f"Failed to parse XML. Raw data saved to {out_err}")
        print(f"Error: {e}")
        return None 
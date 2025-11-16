# fcs_get_table.py

import json
from datagate_client import fetch_data

GETTABLE_URL = "https://api.omnicoll.net/datagate/api/gettable.ashx"

def add_arguments(parser):
    # Add mode-specific args here, if any
    pass

async def run(args):
    params = {}
    data = await fetch_data(params, GETTABLE_URL)
    out_filename = f"fcs_gettable_data.json"
    try:
        parsed = json.loads(data)
        with open(out_filename, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2)
        print(f"GetTable JSON output saved to {out_filename}")
    except Exception:
        with open(out_filename, "w", encoding="utf-8") as f:
            f.write(data)
        print(f"GetTable raw output saved to {out_filename}")

# grabber.py

import os
import argparse
import asyncio
import sys  
import fcs_loggers
import fcs_recordings
import fcs_get_recordings
import fcs_get_table
import fcs_get_correlations

# Add parent directory to Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import SENSOR_DATA, LOGS_DIR

# Mapping of mode to modules
MODE_MODULES = {
    "LOGGERS": fcs_loggers,
    "RECORDINGS": fcs_recordings,
    "GET_RECORDINGS": fcs_get_recordings,
    "GET_TABLE": fcs_get_table,
    "GET_CORRELATIONS": fcs_get_correlations
}

def parse_args():
    # Step 0: Inject default mode if none provided
    if len(sys.argv) == 1:
        sys.argv.extend(["--mode", "LOGGERS"])

    # Step 1: Pre-parse just the --mode argument to choose subparser
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "-m", "--mode",
        choices=MODE_MODULES.keys(),
        required=True,
        help="Which endpoint/mode to use"
    )
    known_args, _ = base_parser.parse_known_args()

    # Step 2: Build full parser for the selected mode
    full_parser = argparse.ArgumentParser(description="Datagate ETL Utility")
    full_parser.add_argument(
        "-m", "--mode",
        choices=MODE_MODULES.keys(),
        required=True,
        help="Which endpoint/mode to use"
    )

    # Add mode-specific arguments
    selected_module = MODE_MODULES[known_args.mode]
    selected_module.add_arguments(full_parser)

    return full_parser.parse_args()

async def main():
    args = parse_args()
    module = MODE_MODULES[args.mode]
    await module.run(args)
    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())

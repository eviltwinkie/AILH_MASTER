import os
import sys
import soundfile as sf
import csv

# === USER CONFIGURATION ===
input_dir = r"..\..\LEAKDATASETS\Dataset\Hydrophone"
output_dir = r"..\..\LEAKDATASETS\DATASET_OUTPUT"

def format_folder_for_output(s):
    """
    Converts any string to uppercase and replaces spaces with hyphens (for folders).
    """
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return s.replace(" ", "-").upper()

def format_file_for_output(s):
    """
    Converts any string to uppercase and replaces spaces with underscores (for files).
    """
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return s.replace(" ", "_").upper()

# Check if input_dir exists
if not os.path.isdir(input_dir):
    print(f"[✗] ERROR: Input directory does not exist: {os.path.abspath(input_dir)}")
    sys.exit(1)

# Ensure output_dir exists
os.makedirs(output_dir, exist_ok=True)
print(f"[✓] Input directory found: {os.path.abspath(input_dir)}")
print(f"[✓] Output directory ready: {os.path.abspath(output_dir)}")

# RAW audio parameters
channels = 1
samplerate = 8000
subtype = 'PCM_32'
endian = 'LITTLE'

# Prepare manifest CSV path
manifest_path = os.path.join(output_dir, "CONVERSION_MANIFEST.CSV")  # Uppercase

header = [
    "SOURCE_PATH", "OUTPUT_PATH", "N_SAMPLES", "CHANNELS", "SAMPLERATE", "DURATION_SEC"
]

def format_cell(cell):
    """Format a manifest cell according to requirements."""
    if isinstance(cell, str):
        return cell.replace(" ", "_").upper()
    elif isinstance(cell, (int, float)):
        return str(cell)
    else:
        return str(cell).replace(" ", "_").upper()

with open(manifest_path, "w", newline='') as manifest_file:
    writer = csv.writer(manifest_file)
    writer.writerow(header)

    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if not filename.lower().endswith('.raw'):
                continue

            # Compute relative path from input_dir, split for multi-level folder formatting
            rel_path = os.path.relpath(root, input_dir)
            if rel_path == ".":
                rel_path_formatted = ""
            else:
                rel_path_formatted = os.path.join(
                    *[format_folder_for_output(part) for part in rel_path.split(os.sep)]
                )

            output_subdir = os.path.join(output_dir, rel_path_formatted)
            os.makedirs(output_subdir, exist_ok=True)

            # Prepare and format output filename
            name_no_ext = os.path.splitext(filename)[0]
            outname = format_file_for_output(name_no_ext) + ".WAV"

            inpath = os.path.join(root, filename)
            outpath = os.path.join(output_subdir, outname)

            try:
                data, _ = sf.read(
                    inpath,
                    channels=channels,
                    samplerate=samplerate,
                    subtype=subtype,
                    endian=endian,
                    format='RAW'
                )

                sf.write(outpath, data, samplerate, subtype=subtype)

                n_samples = len(data)
                duration_sec = n_samples / samplerate

                row = [
                    os.path.abspath(inpath),
                    os.path.abspath(outpath),
                    n_samples,
                    channels,
                    samplerate,
                    duration_sec
                ]
                writer.writerow([format_cell(cell) for cell in row])

                print(f"[✓] {inpath} -> {outpath} ({n_samples} samples, {duration_sec:.2f} s)")
            except Exception as e:
                print(f"[✗] Failed to convert {inpath}: {e}")

print(f"\n✅ ALL FILES PROCESSED. METADATA SAVED TO: {manifest_path}")

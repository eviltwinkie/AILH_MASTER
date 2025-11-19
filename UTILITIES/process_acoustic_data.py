import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import soundfile as sf
import scipy.signal
import csv
from datetime import datetime
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.join("..\..")
ONLINESRC_DIR = os.path.join(BASE_DIR, "DATASET_ONLINESRC")
ACOUSTIC_DATA_DIR = os.path.join(ONLINESRC_DIR, "acoustic_data")
RAW_DATA_DIR = os.path.join(ACOUSTIC_DATA_DIR, "recordings")
PROCESSED_DATA_DIR = os.path.join(ACOUSTIC_DATA_DIR, "PROCESSED")
LABELS_CSV = os.path.join(ACOUSTIC_DATA_DIR, "labelled_acoustic_logger_leaks.csv")
GAIN = 100  # Hardcoded as requested

def find_wav_files(folder):
    return [
        os.path.join(root, f)
        for root, _, files in os.walk(folder)
        for f in files if f.lower().endswith(".wav")
    ]

def resample_audio(y, orig_sr, target_sr):
    """Resample audio array y from orig_sr to target_sr."""
    if orig_sr == target_sr:
        return y
    n_target = int(len(y) * target_sr / orig_sr)
    y_rs = scipy.signal.resample(y, n_target)
    return y_rs

def load_labels_csv(csv_path):
    """
    Load labels from CSV into a dictionary using the 'file' column for filenames.
    Also stores other fields for filename construction.
    """
    labels_dict = {}
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Normalize filename for matching: base filename (no extension)
            base = os.path.splitext(os.path.basename(row['file']))[0]
            labels_dict[base] = {
                'leak_found': row.get('leak_found', ''),
                'noise': row.get('noise', ''),
                'spread': row.get('spread', ''),
                'repaired_as': row.get('repaired_as', ''),
                'datetime': row.get('datetime', ''),
                'siteid': row.get('siteid', ''),
                'recording_id': row.get('recording_id', ''),
            }
    return labels_dict

def parse_datetime_string(dt_str):
    """Convert datetime string '2018-12-16 04:30:00' to '2018_12_16_04_30_00'."""
    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%Y_%m_%d_%H_%M_%S")
    except Exception:
        # If parsing fails, return an empty string or original
        return dt_str.replace("-", "_").replace(":", "_").replace(" ", "_")

def sanitize_filename_component(s):
    """Replace any invalid filename characters with underscores."""
    return re.sub(r'[\\/:"*?<>|]+', '_', s or '')

def map_repaired_as(value):
    """
    Maps the repaired_as field to the specified set.
    """
    mapping = {
        "Boundary Box": "unknown",
        "Ferrule": "ferrule",
        "Fire Hydrant": "firehydrant",
        "Main": "main",
        "Manifold": "mainifold",
        "N/A": "unknown",
        "Network Valve": "valve",
        "OSV": "osv",
        "Repair": "unknown",
        "Service Pipe": "pipe",
        "Washout": "washout"
    }
    if not value or value.strip() == "":
        return "unknown"
    clean_val = value.strip()
    return mapping.get(clean_val, "unknown")

def make_new_filename(label_entry, part_num=None, padded=False):
    """
    Create new filename in the format:
    leak_status~siteid~recording_id~datetime~gain~noise~spread~repaired_as.wav
    Optionally add _part{n} or _padded if needed.
    If noise or spread is blank or -1, set it to 0.
    """
    # Determine leak or noleak
    leak_found = (label_entry.get('leak_found', '') or '').strip().lower()
    if leak_found == "yes":
        leak_status = "leak"
    else:
        leak_status = "noleak"

    siteid = sanitize_filename_component(label_entry.get('siteid', ''))
    recording_id = sanitize_filename_component(label_entry.get('recording_id', ''))
    dt_fmt = sanitize_filename_component(parse_datetime_string(label_entry.get('datetime', '')))
    gain = str(GAIN)

    # Noise and spread: set to "0" if blank or -1
    def clean_numeric(val):
        val = (val or "").strip()
        return "0" if val == "" or val == "-1" else sanitize_filename_component(val)

    noise = clean_numeric(label_entry.get('noise', ''))
    spread = clean_numeric(label_entry.get('spread', ''))

    repaired_as = map_repaired_as(label_entry.get('repaired_as', ''))
    fname = f"{leak_status}~{siteid}~{recording_id}~{dt_fmt}~{gain}~{noise}~{spread}~{repaired_as}"
    if part_num is not None:
        fname += f"_part{part_num}"
    if padded:
        fname += "_padded"
    fname += ".wav"
    return fname, leak_status  # Also return leak_status for folder

def process_one_wav(
    wav_path, input_root, output_root, sample_rate, max_samples, labels_dict
):
    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != sample_rate:
        print(f"[i] {wav_path} has sample rate {sr}, resampling to {sample_rate} ...")
        y = resample_audio(y, sr, sample_rate)
        sr = sample_rate
    total_samples = len(y)
    orig_base = os.path.splitext(os.path.basename(wav_path))[0]
    lookup_base = orig_base.split('_part')[0].replace('_padded', '')

    label_entry = labels_dict.get(lookup_base, None) if labels_dict is not None else None

    results = []
    def print_label_info(label_entry, lookup_base):
        if label_entry is not None:
            mapped_repaired_as = map_repaired_as(label_entry['repaired_as'])
            print(f"  [label] {lookup_base}: leak_found={label_entry['leak_found']}, noise={label_entry['noise']}, spread={label_entry['spread']}, repaired_as={mapped_repaired_as}")
        else:
            print(f"  [label] {lookup_base}: No label info found.")

    if total_samples == max_samples:
        # File is exactly 10s; process as before.
        if label_entry is not None:
            new_fname, leak_status = make_new_filename(label_entry)
        else:
            new_fname = orig_base + ".wav"
            leak_status = "noleak"
        dest_dir = os.path.join(output_root, leak_status.upper())
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, new_fname)
        if os.path.exists(dest_path):
            print(f"[✓] Exists, skipping: {dest_path}")
        else:
            sf.write(dest_path, y, sample_rate)
            print(f"    Copied {wav_path} to {dest_path} ({len(y)} samples)")
        results.append((dest_path, None))
        print_label_info(label_entry, lookup_base)

    elif total_samples > max_samples:
        # File is longer than 10s, so chunk start and end.
        print(f"[i] Extracting first and last 10s from {wav_path} ({total_samples} samples)...")

        # First 10s
        first_chunk = y[:max_samples]
        # Last 10s
        last_chunk = y[-max_samples:]

        for idx, chunk in enumerate([first_chunk, last_chunk], 1):
            if label_entry is not None:
                chunk_fname, leak_status = make_new_filename(label_entry, part_num=idx, padded=True)
            else:
                chunk_fname = f"{orig_base}_part{idx}_padded.wav"
                leak_status = "noleak"
            chunk_dir = os.path.join(output_root, leak_status.upper())
            os.makedirs(chunk_dir, exist_ok=True)
            chunk_out_path = os.path.join(chunk_dir, chunk_fname)
            if os.path.exists(chunk_out_path):
                print(f"[✓] Exists, skipping: {chunk_out_path}")
            else:
                sf.write(chunk_out_path, chunk, sample_rate)
                print(f"    Saved {chunk_out_path} ({len(chunk)} samples)")
            results.append((chunk_out_path, None))
            print_label_info(label_entry, lookup_base)

    elif total_samples < max_samples:
        # File is shorter than 10s, repeat the data and tag "repeat"
        y = np.asarray(y)
        samples_needed = max_samples - total_samples
        repeats = (samples_needed // total_samples) + 1
        repeated = np.tile(y, repeats)[:samples_needed]
        y_repeated = np.concatenate([y, repeated])

        if label_entry is not None:
            repeat_fname, leak_status = make_new_filename(label_entry, padded=False)
            # Manually tag as repeat
            repeat_fname = repeat_fname.replace(".wav", "_repeat.wav")
        else:
            repeat_fname = orig_base + "_repeat.wav"
            leak_status = "noleak"
        repeat_dir = os.path.join(output_root, leak_status.upper())
        os.makedirs(repeat_dir, exist_ok=True)
        repeat_out_path = os.path.join(repeat_dir, repeat_fname)
        if os.path.exists(repeat_out_path):
            print(f"[✓] Exists, skipping: {repeat_out_path}")
        else:
            sf.write(repeat_out_path, y_repeated, sample_rate)
            print(f"    Saved {repeat_out_path} ({len(y_repeated)} samples)")
        results.append((repeat_out_path, None))
        print_label_info(label_entry, lookup_base)

    return results

def standardize_wav_lengths_to_processed(input_root, output_root, sample_rate=4096, max_sec=10, labels_dict=None, max_workers=8):
    """
    Multithreaded version. Output files are split into output_root/LEAK or output_root/NOLEAK folders.
    """
    max_samples = int(sample_rate * max_sec)
    processed_files = []

    wav_files = find_wav_files(input_root)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_one_wav,
                wav_path, input_root, output_root, sample_rate, max_samples, labels_dict
            )
            for wav_path in wav_files
        ]
        for f in as_completed(futures):
            for path, _ in f.result():
                processed_files.append(path)

    return processed_files

if __name__ == "__main__":
    # Load label CSV
    if os.path.exists(LABELS_CSV):
        labels_dict = load_labels_csv(LABELS_CSV)
        print(f"[i] Loaded {len(labels_dict)} label entries from {LABELS_CSV}")
    else:
        print(f"[!] Label CSV not found at {LABELS_CSV}. Label printing will be skipped.")
        labels_dict = None

    processed_files = standardize_wav_lengths_to_processed(
        input_root=RAW_DATA_DIR,
        output_root=PROCESSED_DATA_DIR,
        sample_rate=4096,
        max_sec=10,
        labels_dict=labels_dict,
        max_workers=8  # Adjust for your system
    )

    print(f"[i] Processed {len(processed_files)} files to {PROCESSED_DATA_DIR} (split by LEAK/NOLEAK).")
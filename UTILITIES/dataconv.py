import os
import numpy as np
import soundfile as sf
import librosa

# === CONFIGURATION ===
input_dir = r"Hydrophone"    # Directory containing .raw files (update as needed)
output_base = "converted_wav_4096_segments"  # Output directory for processed WAV files
os.makedirs(output_base, exist_ok=True)      # Ensure output directory exists

# Parameters describing the raw audio file format and processing
channels = 1                 # Number of channels in raw files (mono)
orig_samplerate = 8000       # Original sample rate of raw files (Hz)
target_samplerate = 4096     # Target sample rate after resampling (Hz)
subtype = 'PCM_32'           # Raw PCM data, 32-bit samples
endian = 'LITTLE'            # Little-endian byte order
segment_sec = 10             # Desired segment length in seconds
segment_len = target_samplerate * segment_sec  # Segment length in samples

count = 0                    # Counter for processed segments

# === MAIN PROCESSING LOOP ===
# Recursively search through input_dir for files to process
for root, dirs, files in os.walk(input_dir):
    for filename in files:
        # Skip non-raw files
        if not filename.lower().endswith('.raw'):
            continue
        inpath = os.path.join(root, filename)
        print(f"Processing: {inpath}")
        try:
            # --- Read RAW file as audio ---
            # soundfile.read can read RAW data if parameters are specified
            data, _ = sf.read(
                inpath,
                channels=channels,
                samplerate=orig_samplerate,
                subtype=subtype,
                endian=endian,
                format='RAW'
            )

            # --- Resample audio from original to target sample rate (e.g., 8000 Hz to 4096 Hz) ---
            data = librosa.resample(
                data.astype(np.float32), 
                orig_sr=orig_samplerate, 
                target_sr=target_samplerate
            )

            # --- Segment the audio into non-overlapping 10-second windows ---
            num_segments = int(np.ceil(len(data) / segment_len))  # Number of 10s segments

            # Prepare output subdirectory structure to mirror input
            rel_subdir = os.path.relpath(root, input_dir)
            output_dir = os.path.join(output_base, rel_subdir)
            os.makedirs(output_dir, exist_ok=True)
            base = os.path.splitext(filename)[0]  # Base filename without extension

            for i in range(num_segments):
                start = i * segment_len
                end = start + segment_len
                segment = data[start:end]

                # If last segment is too short, pad it with zeros (silence)
                if len(segment) < segment_len:
                    segment = np.pad(segment, (0, segment_len - len(segment)))

                # Create output filename indicating segment number and processing parameters
                outname = f"{base}_seg{i+1:03}_4096hz_10s.wav"
                outpath = os.path.join(output_dir, outname)

                # Save the segment as a WAV file
                sf.write(outpath, segment, target_samplerate)
                print(f"  Saved: {outpath}")
                count += 1

        except Exception as e:
            # Print error message if any file can't be read or processed
            print(f"Could not read {inpath}: {e}")

print(f"\n✅ Processed {count} segments. Files saved in: {output_base}")

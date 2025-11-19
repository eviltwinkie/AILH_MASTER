import os
import numpy as np
import soundfile as sf
from pydub import AudioSegment

input_dir = "voice_samples"            # Your input folder with MP3 or WAVs
output_dir = "voice_samples_wav4096"   # Output folder
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.lower().endswith(('.mp3', '.wav')):
        continue
    inpath = os.path.join(input_dir, filename)
    print(f"Processing {inpath}...")
    # Use pydub to load ANY format
    audio = AudioSegment.from_file(inpath).set_frame_rate(4096).set_channels(1)
    data = np.array(audio.get_array_of_samples()).astype(np.float32) / (2**15)
    N = 4096 * 10  # samples for 10 seconds
    # Trim or pad to exactly 10 seconds
    if len(data) > N:
        data = data[:N]
    elif len(data) < N:
        data = np.pad(data, (0, N - len(data)))
    base = os.path.splitext(filename)[0]
    outname = os.path.join(output_dir, f"{base}_4096hz_10s.wav")
    sf.write(outname, data, 4096)
    print(f"Saved: {outname}")

print("\n✅ All done — processed files are in:", output_dir)

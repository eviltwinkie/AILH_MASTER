from old_config import SAMPLE_RATE, CNN_BATCH_SIZE, MAX_THREAD_WORKERS, N_FFT, HOP_LENGTH, N_MELS, LONG_TERM_SEC, SHORT_TERM_SEC, STRIDE_SEC, CACHE_DIR, MAX_THREAD_WORKERS, TMPDIR

import h5py
import time
import numpy as np
import torch
from torch import multiprocessing as torch_mp
import torchaudio
import soundfile as sf
from tqdm import tqdm
import tensorflow as tf
import concurrent.futures
import os
import gc
import sys
import glob
import pickle

from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from queue import Queue
from threading import Thread
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, as_completed


def after_trial_cleanup():
    """Lightweight version for Optuna trial cleanup."""
    print("[i] Cleaning up GPU and CPU memory after trial...")
    try:
        tf.keras.backend.clear_session()
    except Exception as e:
        print(f"[!] TF session cleanup failed: {e}")
    try:
        gc.collect()
    except Exception as e:
        print(f"[!] Python GC failed: {e}")
    torch.cuda.empty_cache()
    sys.stdout.flush()
    sys.stderr.flush()
    time.sleep(0.2)

def compute_max_poolings(freq_dim, time_dim, min_freq_dim=8, min_time_dim=1):
    f, t = freq_dim, time_dim
    poolings = 0
    while f >= min_freq_dim*2 and t >= min_time_dim*2:
        f = (f - 2) // 2 + 1 if f >= 2 else f
        t = (t - 2) // 2 + 1 if t >= 2 else t
        poolings += 1
    return poolings


# def process_segments_to_mel_memmap(
#     all_segments, sr, n_fft, hop_length, n_mels, batch_size=128, device='cuda', memmap_path=None
# ):
#     all_segments = np.stack([s.astype(np.float32) for s in all_segments])
#     seg_tensor = torch.from_numpy(all_segments).float()
#     del all_segments
#     gc.collect()
#     num_batches = int(np.ceil(len(seg_tensor) / batch_size))
#     first_batch = batch_mel_spectrogram(
#         seg_tensor[:batch_size].to(device), sr, n_fft, hop_length, n_mels, device=device
#     )
#     batch_shape = first_batch.cpu().numpy().astype(np.float32).shape
#     n_segments = len(seg_tensor)
#     n_mels, frames = batch_shape[1], batch_shape[2]
#     if memmap_path:
#         mels = np.memmap(memmap_path, dtype=np.float32, mode='w+', shape=(n_segments, n_mels, frames))
#     else:
#         mels = np.empty((n_segments, n_mels, frames), dtype=np.float32)
#     mels[:batch_size] = first_batch.cpu().numpy().astype(np.float32)
#     bar = tqdm(range(1, num_batches), desc="[MEL] Processing")
#     for i in bar:
#         start = i * batch_size
#         end = min((i+1)*batch_size, n_segments)
#         batch = seg_tensor[start:end].to(device)
#         m = batch_mel_spectrogram(batch, sr, n_fft, hop_length, n_mels, device=device)
#         mels[start:end] = m.cpu().numpy().astype(np.float32)
#         del batch, m
#         torch.cuda.empty_cache()
#         gc.collect()
#     del seg_tensor
#     gc.collect()
#     if memmap_path:
#         mels.flush()
#     sys.stdout.flush()
#     sys.stderr.flush()
#     print(f"[✓] Processed {len(mels)} segments to Mels with shape {mels.shape} (memmap: {memmap_path is not None})")
#     return mels

# def load_all_segments_two_stage_parallel(wav_paths, sample_rate=SAMPLE_RATE, long_term_sec=LONG_TERM_SEC, short_term_sec=SHORT_TERM_SEC, stride_sec=STRIDE_SEC, max_workers=MAX_THREAD_WORKERS):
#     segments_per_wav = [None] * len(wav_paths)
#     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = {
#             executor.submit(
#                 extract_two_stage_segments_from_wav,
#                 wav_path, sample_rate,
#                 long_term_sec, short_term_sec, stride_sec
#             ): idx
#             for idx, wav_path in enumerate(wav_paths)
#         }
#         for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Extracting Segments"):
#             idx = futures[future]
#             try:
#                 segments_per_wav[idx] = future.result()
#             except Exception as e:
#                 print(f"[THREAD ERROR] Failed processing file idx={idx}: {e}")
#                 segments_per_wav[idx] = []
#         executor.shutdown(wait=True)
#     sys.stdout.flush()
#     sys.stderr.flush()
#     gc.collect()
#     return segments_per_wav





# def extract_segments_generator(wav_files, labels, sample_rate, long_term_sec, short_term_sec, stride_sec, batch_wavs=128):
#     """Yield (batch_segments, batch_labels) for each batch of files."""
#     for i in range(0, len(wav_files), batch_wavs):
#         batch_files = wav_files[i:i+batch_wavs]
#         batch_labels = labels[i:i+batch_wavs]
#         segments, seg_labels = [], []
#         for wav_path, label in zip(batch_files, batch_labels):
#             try:
#                 data, sr = sf.read(wav_path)
#                 if sr != sample_rate: continue
#                 long_len = int(long_term_sec * sample_rate)
#                 short_len = int(short_term_sec * sample_rate)
#                 stride = int(stride_sec * sample_rate)
#                 for lstart in range(0, len(data) - long_len + 1, stride):
#                     long_seg = data[lstart:lstart + long_len]
#                     for sstart in range(0, long_len - short_len + 1, stride):
#                         short_seg = long_seg[sstart:sstart + short_len]
#                         segments.append(short_seg.astype(np.float32))
#                         seg_labels.append(label)
#             except Exception as e:
#                 print(f"[!] Error in {wav_path}: {e}")
#         if segments:
#             yield segments, seg_labels








def batch_mel_spectrogram(
    wavs, sr, n_fft, hop_length, n_mels, fmin=20, fmax=None, device='cuda',
    power=2.0, norm='slaney', mel_scale='htk'
):
    """
    Computes Mel spectrograms for a batch of waveforms using TorchAudio.

    Args:
        wavs (torch.Tensor): (batch, samples) float32, on CPU or GPU.
        sr (int): Sample rate.
        n_fft (int): FFT window size.
        hop_length (int): Hop size.
        n_mels (int): Number of mel bins.
        fmin (int): Minimum frequency.
        fmax (int): Maximum frequency. If None, set to sr // 2.
        device (str): 'cuda' or 'cpu'.

    Returns:
        torch.Tensor: (batch, n_mels, frames) float32, on device.
    """
    wavs = wavs.to(device)
    fmax = fmax or sr // 2
    if wavs.dim() == 1:
        wavs = wavs.unsqueeze(0)
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=fmin,
        f_max=fmax,
        power=power,
        norm=norm,
        mel_scale=mel_scale
    ).to(device)
    with torch.no_grad():
        mels = mel_spec(wavs)
        mels = 10.0 * torch.log10(mels.clamp(min=1e-8))
    return mels

def wav_segment_worker(args):
    wav_path, label, sample_rate, long_term_sec, short_term_sec, stride_sec = args
    import soundfile as sf
    try:
        data, sr = sf.read(wav_path)
        if sr != sample_rate:
            raise ValueError(f"Sample rate mismatch: {sr} != {sample_rate} for {wav_path}")
        long_len = int(long_term_sec * sample_rate)
        short_len = int(short_term_sec * sample_rate)
        stride = int(stride_sec * sample_rate)
        segments = []
        for lstart in range(0, len(data) - long_len + 1, stride):
            long_seg = data[lstart:lstart + long_len]
            for sstart in range(0, long_len - short_len + 1, stride):
                short_seg = long_seg[sstart:sstart + short_len]
                segments.append(short_seg.astype(np.float32))
        del data
        return segments, label
    except Exception as e:
        print(f"[ERROR] {wav_path}: {e}")
        return [], label

def load_wavs_parallel(
    wav_files, labels, n_mels, n_fft, hop_length, sample_rate,
    long_term_sec, short_term_sec, stride_sec,
    augment=False, cache_prefix="cache", debug_plot_first=False,
    max_workers=MAX_THREAD_WORKERS, device='cuda', CACHE_DIR="/mnt/d/AILH_CACHE", TMP_DIR="/mnt/d/AILH_TMP",
    batch_size=512, queue_size=4, **kwargs
):
    def hash_files(files):
        return abs(hash("".join(files))) % (10**8)
    data_hash = hash_files(wav_files)
    cache_X = os.path.join(CACHE_DIR, f"{cache_prefix}_X_{data_hash}.h5")
    cache_le = os.path.join(CACHE_DIR, f"{cache_prefix}_le_{data_hash}.pkl")
    cache_maxframes = os.path.join(CACHE_DIR, f"{cache_prefix}_maxframes_{data_hash}.txt")
    tmp_h5_path = os.path.join(TMP_DIR, "tmp_mels_stream.h5")

    def cleanup_old_cache(pattern, current_file):
        import glob
        for f in glob.glob(pattern):
            if f != current_file:
                try:
                    os.remove(f)
                    print(f"[i] Removed old cache: {os.path.basename(f)}")
                except OSError:
                    pass
    cleanup_old_cache(os.path.join(CACHE_DIR, f"{cache_prefix}_X_*.h5"), cache_X)
    cleanup_old_cache(os.path.join(CACHE_DIR, f"{cache_prefix}_le_*.pkl"), cache_le)
    cleanup_old_cache(os.path.join(CACHE_DIR, f"{cache_prefix}_maxframes_*.txt"), cache_maxframes)

    if os.path.exists(cache_X) and os.path.exists(cache_le) and os.path.exists(cache_maxframes):
        print(f"[i] Loading cached Mel spectrograms from {cache_X} ...")
        h5 = h5py.File(cache_X, 'r')
        X = h5['X']
        Y = h5['Y_onehot']
        with open(cache_le, "rb") as f_le:
            le = pickle.load(f_le)
        with open(cache_maxframes, "r") as f:
            max_frames = int(f.read())
        return X, Y, le, max_frames

    # --- Pass 1: Count segments and max_frames
    print("[✓] Counting total segments and max mel frames...")
    total_segments, max_frames = 0, 0
    worker_args = [(wav_files[i], labels[i], sample_rate, long_term_sec, short_term_sec, stride_sec) for i in range(len(wav_files))]
    from multiprocessing import get_context
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context("spawn")) as pool:
        for segs, _ in tqdm(pool.map(wav_segment_worker, worker_args), total=len(worker_args), desc="Counting"):
            total_segments += len(segs)
            for seg in segs:
                frames = 1 + (len(seg) - n_fft) // hop_length
                if frames > max_frames:
                    max_frames = frames
            del segs
    print(f"[✓] {total_segments} total segments, {max_frames} max mel frames.")
    gc.collect()

    # --- Allocate HDF5
    h5 = h5py.File(tmp_h5_path, 'w')
    chunk = min(batch_size, total_segments)
    X_dset = h5.create_dataset(
        'X', shape=(total_segments, n_mels, max_frames, 1), maxshape=(None, n_mels, max_frames, 1),
        dtype='float32', chunks=(chunk, n_mels, max_frames, 1), compression='lzf'
    )
    Y_dset = h5.create_dataset(
        'Y', shape=(total_segments,), dtype='i4', chunks=(chunk,)
    )
    segment_counter = [0]
    le = LabelEncoder(); le.fit(labels)
    with open(cache_le, "wb") as f_le: pickle.dump(le, f_le)

    # --- GPU writer (consumer)
    def gpu_writer(batch_queue):
        while True:
            item = batch_queue.get()
            if item is None: break
            batch_segments, batch_labels = item
            if not batch_segments: continue
            seg_tensor = torch.from_numpy(np.stack(batch_segments)).float()
            with torch.no_grad():
                mels = batch_mel_spectrogram(seg_tensor, sample_rate, n_fft, hop_length, n_mels, device=device)
                mels = mels.cpu().numpy().astype(np.float32)
            for i in range(len(mels)):
                mel = mels[i]
                n_frames = mel.shape[1]
                pad_width = max_frames - n_frames
                if pad_width < 0:
                    mel_arr = mel[:, :max_frames]
                elif pad_width > 0:
                    mel_arr = np.pad(mel, ((0,0),(0,pad_width)), mode='constant', constant_values=np.min(mel))
                else:
                    mel_arr = mel
                X_dset[segment_counter[0], :, :, 0] = mel_arr
                Y_dset[segment_counter[0]] = le.transform([batch_labels[i]])[0]
                segment_counter[0] += 1
            del seg_tensor, mels, batch_segments, batch_labels, mel_arr
            torch.cuda.empty_cache(); gc.collect()
            h5.flush()
        h5.flush()

    batch_queue = Queue(maxsize=queue_size)
    writer_thread = Thread(target=gpu_writer, args=(batch_queue,), daemon=True)
    writer_thread.start()

    # --- Streaming Extraction+Batching
    print("[✓] Streaming extraction and GPU Mel (true streaming)...")
    batch_segments, batch_labels = [], []
    def segment_generator():
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context("spawn")) as pool:
            for segs, label in pool.map(wav_segment_worker, worker_args):
                for seg in segs:
                    yield seg, label

    for seg, label in tqdm(segment_generator(), total=total_segments, desc="Extracting+Mel (streaming)"):
        batch_segments.append(seg)
        batch_labels.append(label)
        if len(batch_segments) >= batch_size:
            batch_queue.put((batch_segments, batch_labels))
            batch_segments, batch_labels = [], []
    if batch_segments:
        batch_queue.put((batch_segments, batch_labels))
    batch_queue.put(None)
    writer_thread.join()
    h5.flush()
    gc.collect()

    # One-hot labels, finish HDF5, cleanup
    Y_onehot = to_categorical(Y_dset[()], num_classes=len(le.classes_))
    if 'Y_onehot' in h5:
        del h5['Y_onehot']
    h5.create_dataset('Y_onehot', data=Y_onehot, dtype='float32')
    h5.flush()
    with open(cache_maxframes, "w") as f: f.write(str(max_frames))
    h5.close()
    os.rename(tmp_h5_path, cache_X)
    print(f"[i] Saved Mel spectrograms HDF5 to {cache_X}")

    h5 = h5py.File(cache_X, 'r')
    X = h5['X']; Y = h5['Y_onehot']
    le = pickle.load(open(cache_le, "rb"))
    max_frames = int(open(cache_maxframes).read())
    return X, Y, le, max_frames
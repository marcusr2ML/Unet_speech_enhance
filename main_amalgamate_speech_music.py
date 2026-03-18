import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.io import wavfile

# ============================================================
# CONFIG
# ============================================================

SOURCE_ROOT = "/home/marc/Downloads/test-clean/LibriTTS/test-clean"
DEST_ROOT = "/home/marc/Downloads/speech_music"

sample_rate = 16000  # Your training rate

os.makedirs(DEST_ROOT, exist_ok=True)


# ============================================================
# LOAD WAV
# ============================================================

def load_and_process(FILE_PATH):
    try:
        sr, data = wavfile.read(FILE_PATH)

        # Convert stereo → mono
        if data.ndim > 1:
            data = data[:, 0]

        # Normalize input dtype
        audio_np = data.astype(np.float32) / np.iinfo(data.dtype).max

        # Resample if needed
        if sr != sample_rate:
            num = int(audio_np.shape[0] * (sample_rate / sr))
            audio_np = np.interp(
                np.linspace(0, 1, num, endpoint=False),
                np.linspace(0, 1, audio_np.shape[0], endpoint=False),
                audio_np
            )
            print(f"Resampled {FILE_PATH} → {sample_rate}")

        # Final normalize
        audio_final = audio_np / (np.max(np.abs(audio_np)) + 1e-9)
        return audio_final

    except Exception as e:
        print(f"Error loading {FILE_PATH}: {e}")
        return None


# ============================================================
# EXTRACT ALL WAV FILES
# ============================================================

clean_idx = 0

for root, dirs, files in os.walk(SOURCE_ROOT):
    for f in files:
        if f.lower().endswith(".wav"):
            in_path = os.path.join(root, f)

            audio = load_and_process(in_path)
            if audio is None:
                continue

            out_path = os.path.join(DEST_ROOT, f"clean_{clean_idx:04d}.wav")

            wavfile.write(out_path, sample_rate, (audio * 32767).astype(np.int16))

            print(f"Saved: {out_path}")
            clean_idx += 1

print(f"\nDONE! Extracted {clean_idx} speech clips.")


 
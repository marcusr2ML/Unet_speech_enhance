import os
import numpy as np
from scipy.io import wavfile

# ============================================================
# CONFIG
# ============================================================

SOURCE_ROOT = "/home/marc/Downloads/speech_music"
DEST_ROOT   = "/home/marc/Downloads/speech_music_normalized"

TARGET_DB = -25.0   # Target loudness for all clips
sample_rate = 16000

os.makedirs(DEST_ROOT, exist_ok=True)

# ============================================================
# UTILITIES
# ============================================================

def rms_db(wav):
    wav = np.asarray(wav, dtype=np.float32)
    rms = np.sqrt(np.mean(wav**2) + 1e-12)
    db = 20 * np.log10(rms)
    return db

def normalize_to_db(wav, target_db=TARGET_DB):
    current_db = rms_db(wav)
    diff = target_db - current_db
    factor = 10 ** (diff / 20)
    return wav * factor

def load_and_process(FILE_PATH, target_sr=sample_rate):
    try:
        sr, data = wavfile.read(FILE_PATH)
        if data.ndim > 1:
            data = data[:, 0]
        audio_np = data.astype(np.float32) / np.iinfo(data.dtype).max

        # Resample if needed
        if sr != target_sr:
            num = int(audio_np.shape[0] * (target_sr / sr))
            audio_np = np.interp(
                np.linspace(0, 1, num, endpoint=False),
                np.linspace(0, 1, audio_np.shape[0], endpoint=False),
                audio_np
            )

        return audio_np
    except Exception as e:
        print(f"Error loading {FILE_PATH}: {e}")
        return None

# ============================================================
# PROCESS ALL WAV FILES
# ============================================================

file_list = [f for f in os.listdir(SOURCE_ROOT) if f.lower().endswith(".wav")]

for idx, f in enumerate(file_list):
    in_path = os.path.join(SOURCE_ROOT, f)
    audio = load_and_process(in_path)
    if audio is None:
        continue

    # Normalize loudness
    audio = normalize_to_db(audio, target_db=TARGET_DB)

    # Final peak normalize to [-1,1]
    audio = audio / (np.max(np.abs(audio)) + 1e-9)

    # Save
    out_path = os.path.join(DEST_ROOT, f)
    wavfile.write(out_path, sample_rate, (audio * 32767).astype(np.int16))

    print(f"{idx+1}/{len(file_list)}: {f} → normalized to {TARGET_DB} dB")

print("\nDONE! All clips normalized to the same loudness.")
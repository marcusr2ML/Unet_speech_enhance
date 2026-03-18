import os
import glob
import numpy as np
from scipy.io import wavfile

# ============================================================
# CONFIG
# ============================================================

SPEECH_ROOT = "/home/marc/Downloads/speech_music_normalized"
MUSIC_ROOT  = "/home/marc/Downloads/train_music/test"
OUTPUT_ROOT = "/home/marc/Downloads/train_music/speech_mixture_dataset"

sample_rate = 16000

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ============================================================
# UTILITIES
# ============================================================

def load_and_process(FILE_PATH, target_sr=sample_rate):
    try:
        sr, data = wavfile.read(FILE_PATH)
        if data.ndim > 1:
            data = data[:, 0]

        audio_np = data.astype(np.float32) / np.iinfo(data.dtype).max

        if sr != target_sr:
            num = int(audio_np.shape[0] * (target_sr / sr))
            audio_np = np.interp(
                np.linspace(0, 1, num, endpoint=False),
                np.linspace(0, 1, audio_np.shape[0], endpoint=False),
                audio_np
            )

        # normalize peak to [-1,1]
        audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-9)
        return audio_np
    except Exception as e:
        print(f"Error loading {FILE_PATH}: {e}")
        return None

# ============================================================
# LOAD FILE LISTS
# ============================================================

# Speech clips (directly in folder)
speech_files = sorted([os.path.join(SPEECH_ROOT, f) 
                       for f in os.listdir(SPEECH_ROOT) 
                       if f.lower().endswith(".wav")])

# Music clips (recursively find mixture.wav)
music_files = sorted(glob.glob(os.path.join(MUSIC_ROOT, "**", "mixture.wav"), recursive=True))

if not speech_files:
    raise ValueError("No speech files found!")
if not music_files:
    raise ValueError("No music files found!")

# ============================================================
# CREATE DATASET
# ============================================================

speech_idx = 0
music_idx  = 0
mixture_count = 1

while speech_idx < len(speech_files):
    # Load music clip
    music_path = music_files[music_idx]
    music = load_and_process(music_path)
    if music is None:
        music_idx = (music_idx + 1) % len(music_files)
        continue

    combined_speech = []
    total_speech_len = 0

    # Concatenate speech until music is filled
    while total_speech_len < len(music) and speech_idx < len(speech_files):
        speech_path = speech_files[speech_idx]
        speech_clip = load_and_process(speech_path)
        if speech_clip is None:
            speech_idx += 1
            continue

        combined_speech.append(speech_clip)
        total_speech_len += len(speech_clip)
        speech_idx += 1

    # Create output folder for clean speech alongside mixture
    CLEAN_ROOT = os.path.join(os.path.dirname(OUTPUT_ROOT), "speech_clean_dataset")
    os.makedirs(CLEAN_ROOT, exist_ok=True)
    
    # Concatenate speech
    if combined_speech:
        speech_concat = np.concatenate(combined_speech)
    else:
        speech_concat = np.zeros_like(music)
    
    # Crop to music length if needed
    min_len = min(len(speech_concat), len(music))
    speech_concat = speech_concat[:min_len]
    music = music[:min_len]
    
    # Create mixture
    mixture = music + speech_concat
    mixture = mixture / (np.max(np.abs(mixture)) + 1e-9)  # normalize mixture
    
    # Save files
    mixture_path = os.path.join(OUTPUT_ROOT, f"mixture{mixture_count:03d}.wav")
    clean_path   = os.path.join(OUTPUT_ROOT, f"clean{mixture_count:03d}.wav")
    speech_only_path = os.path.join(CLEAN_ROOT, f"clean{mixture_count:03d}.wav")  # save cropped speech here
    
    wavfile.write(mixture_path, sample_rate, (mixture * 32767).astype(np.int16))
    wavfile.write(clean_path, sample_rate, (speech_concat * 32767).astype(np.int16))
    wavfile.write(speech_only_path, sample_rate, (speech_concat * 32767).astype(np.int16))  # save separately
    
    print(f"Saved {mixture_path} + {clean_path} + {speech_only_path}")
    
    mixture_count += 1
    music_idx = (music_idx + 1) % len(music_files)  # loop music if needed

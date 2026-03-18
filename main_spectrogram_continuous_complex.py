import numpy as np
import torch
import sounddevice as sd
import torch.nn.functional as F
from unet import UNet

# ============================================================
# CONFIG — MUST MATCH TRAINING
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "unet_vocal_SR48k_mask_2.pth"

sample_rate = 44100
N_FFT = 512
HOP_LENGTH = N_FFT // 4
T_target = 256

WINDOW_SIZE = (T_target - 1) * HOP_LENGTH + N_FFT
WINDOW_HOP = WINDOW_SIZE // 4  # overlap

print("WINDOW_SIZE:", WINDOW_SIZE, "→ STFT time dim T:", T_target)

win = torch.hann_window(N_FFT, device=DEVICE)

# ============================================================
# ============================================================
model = UNet(in_channels=2, num_classes=2).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ============================================================
# COMPLEX STFT
# ============================================================
def compute_complex_stft(wave):
    wave = wave.astype(np.float32)
    wave /= np.max(np.abs(wave)) + 1e-9

    t = torch.from_numpy(wave).unsqueeze(0).to(DEVICE)
    S = torch.stft(
        t,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=win,
        return_complex=True,
        center=True
    )  # (1, F, T)

    S_real = S.real
    S_imag = S.imag
    S_complex = torch.cat([S_real, S_imag], dim=0)  # (2, F, T)

    # Crop to multiples of 16 for UNet
    F_new = (S_complex.size(1) // 16) * 16
    T_new = (S_complex.size(2) // 16) * 16
    return S_complex[:, :F_new, :T_new]

# ============================================================
# UNET FORWARD
# ============================================================
def run_unet_complex(S_complex):
    inp = S_complex.unsqueeze(0)  # (1, 2, F, T)
    with torch.no_grad():
        pred = model(inp)
    return pred.squeeze(0)  # (2, F, T)

# ============================================================
# ISTFT
# ============================================================
def istft_from_complex(pred_complex, length):
    target_f = N_FFT // 2 + 1
    current_f = pred_complex.size(1)
    if current_f < target_f:
        diff = target_f - current_f
        pred_complex = F.pad(pred_complex, (0, 0, 0, diff))

    real = pred_complex[0]
    imag = pred_complex[1]
    S = torch.complex(real, imag).unsqueeze(0)

    waveform = torch.istft(
        S,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=win,
        length=length
    )

    return waveform.squeeze().cpu().numpy()

# ============================================================
# LIVE RECORD → FILTER → PLAY LOOP
# ============================================================
if __name__ == "__main__":
    duration = 5  # initial recording seconds
    num_samples = int(duration * sample_rate)

    print("Initial recording...")
    audio_input = sd.rec(num_samples, samplerate=sample_rate, channels=1, blocking=True)
    audio_input = audio_input.flatten()

    try:
        while True:
            # --- CHUNKED PROCESSING ---
            output_audio = np.zeros(len(audio_input))
            window_count = np.zeros(len(audio_input))

            for i in range(0, len(audio_input) - WINDOW_SIZE, WINDOW_HOP):
                chunk = audio_input[i : i + WINDOW_SIZE]

                S_complex = compute_complex_stft(chunk)
                pred_complex = run_unet_complex(S_complex)
                audio_out = istft_from_complex(pred_complex, len(chunk))

                output_audio[i : i + WINDOW_SIZE] += audio_out
                window_count[i : i + WINDOW_SIZE] += 1

            window_count[window_count == 0] = 1
            final_output = output_audio / window_count
            a = .88
            b = 1-a**2
            final_output = (a* final_output + b*audio_input[:len(final_output)])/(a**2+b**2)

            # --- PLAY & RECORD NEXT ---
            print("Playing filtered / recording next...")
            audio_input = sd.playrec(
                final_output,
                samplerate=sample_rate,
                channels=1,
                blocking=True
            ).flatten()

    except KeyboardInterrupt:
        sd.stop()
        print("\nStopped.")
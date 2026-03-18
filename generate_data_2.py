import os
import numpy as np
from scipy.io import wavfile
import torch
from torch.utils.data import Dataset

TARGET_SR = 48000

class GenerateData(Dataset):
    def __init__(self, ROOT,
                 sample_rate=TARGET_SR,
                 N_FFT=1536, HOP_LENGTH=384, WINDOW_SIZE=196608, WINDOW_HOP=49152):
        
        
        self.ROOT = ROOT        
        self.sample_rate = sample_rate
        self.N_FFT = N_FFT
        self.HOP_LENGTH = HOP_LENGTH
        self.WINDOW_SIZE = WINDOW_SIZE
        self.WINDOW_HOP = WINDOW_HOP
        self.win = torch.hann_window(self.N_FFT)



        folders = sorted([
            f for f in os.listdir(ROOT)
            if os.path.isdir(os.path.join(ROOT, f))
            and os.path.exists(os.path.join(ROOT, f, "mixture.wav"))
            and os.path.exists(os.path.join(ROOT, f, "vocals.wav"))
        ])
    
        print("\nFound folders:", folders)
    
        mixtures_waves = []
        vocals_waves = []
    
        for folder in folders:              #normalizes and windows the data
            print("Loading:", folder)
            mix = self.load_and_process(os.path.join(ROOT, folder, "mixture.wav"))
            voc = self.load_and_process(os.path.join(ROOT, folder, "vocals.wav"))
    
            if mix is None or voc is None:
                continue
    
            max_val = max(np.max(np.abs(mix)), np.max(np.abs(voc)))
            mix = mix / (max_val + 1e-9)
            voc = voc / (max_val + 1e-9)
    
            mixtures_waves.extend(self._make_windows(mix))
            vocals_waves.extend(self._make_windows(voc))
            
    
        self.X_stft = [self._compute_complex_stft(w) for w in mixtures_waves]    #precomputed stft to be input into DataLoader
        self.Y_stft = [self._compute_complex_stft(w) for w in vocals_waves]
        print(f"Created {len(mixtures_waves)} training windows")




    # ---------------------------------------------------
    # Audio loading
    # ---------------------------------------------------
    def load_and_process(self, FILE_PATH):
        try:
            sr, data = wavfile.read(FILE_PATH)
    
            if data.ndim > 1:
                data = data[:, 0]
    
            audio_np = data.astype(np.float32) / np.iinfo(data.dtype).max
    
            if sr != self.sample_rate:
                num = int(audio_np.shape[0] * (self.sample_rate / sr))
                audio_np = np.interp(
                    np.linspace(0, 1, num, endpoint=False),
                    np.linspace(0, 1, audio_np.shape[0], endpoint=False),
                    audio_np
                )
                print(f"Resampled {FILE_PATH} → {self.sample_rate}")
    
            return audio_np
    
        except Exception as e:
            print(f"Error loading {FILE_PATH}: {e}")
            return None
    # ============================================================
    # WINDOW CUTTING
    # ============================================================
    
    def _make_windows(self, wave):
        out = []
        L = len(wave)
        i = 0
        while i + self.WINDOW_SIZE <= L:
            out.append(wave[i:i+self.WINDOW_SIZE])
            i += self.WINDOW_HOP
        return out
    # ---------------------------------------------------
    # STFT computation
    # ---------------------------------------------------
    def _compute_complex_stft(self, wave):
        t = torch.from_numpy(wave).float().unsqueeze(0)
        S = torch.stft(t, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH,
                       window=self.win, return_complex=True, center=True)
        S_complex = torch.cat([S.real, S.imag], dim=0)
        F_new = (S_complex.size(1)//16)*16   #padding choosen in accord to the depth of Unet
        T_new = (S_complex.size(2)//16)*16
        return S_complex[:, :F_new, :T_new]


    def __len__(self):
        return len(self.X_stft)

    def __getitem__(self, idx):
        return self.X_stft[idx],self.Y_stft[idx]



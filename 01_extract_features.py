import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────
DATA_DIR   = "data/genres_original"
OUTPUT_CSV = "features.csv"
SR         = 22050   # sample rate
N_MFCC     = 13      # number of MFCC coefficients
HOP_LENGTH = 512     # STFT hop (samples per frame)
N_FFT      = 2048    # FFT window size

GENRES = [
    "blues", "classical", "country", "disco",
    "hiphop", "jazz", "metal", "pop", "reggae", "rock"
]

# ── Feature extraction function ────────────────────────────────────────────
def extract_features(file_path):
    """
    Returns a flat numpy array of audio features for one audio file.

    Features extracted:
      - MFCCs (13)          : timbre / tonal texture
      - MFCC delta (13)     : rate-of-change of timbre
      - Chroma STFT (12)    : harmonic / pitch-class content
      - Spectral centroid(1): brightness (center of mass of spectrum)
      - Spectral rolloff (1): frequency below which 85% of energy falls
      - Spectral bandwidth(1): spread of the spectrum
      - Spectral contrast (7): valley-to-peak difference per sub-band
      - Zero-crossing rate (1): how often signal crosses zero (noisiness)
      - RMS energy (1)      : loudness / dynamic range
      - Tempo (1)           : BPM estimated via beat tracking
      Total: 51 features (mean across time frames)
    """
    try:
        y, sr = librosa.load(file_path, sr=SR, duration=30.0, mono=True)
    except Exception as e:
        print(f"  SKIP {file_path}: {e}")
        return None

    features = []

    # ── STFT magnitude spectrogram (basis for many features) ───────────────
    stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))

    # ── MFCCs ──────────────────────────────────────────────────────────────
    # Convert power spectrogram to mel scale, then apply DCT → MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT,
                                  hop_length=HOP_LENGTH)
    mfcc_delta = librosa.feature.delta(mfcc)   # first derivative
    features.extend(np.mean(mfcc, axis=1))           # 13 values
    features.extend(np.mean(mfcc_delta, axis=1))     # 13 values

    # ── Chroma ─────────────────────────────────────────────────────────────
    # Projects spectrum onto 12 pitch classes (C, C#, D, ... B)
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr, n_fft=N_FFT,
                                          hop_length=HOP_LENGTH)
    features.extend(np.mean(chroma, axis=1))          # 12 values

    # ── Spectral centroid ──────────────────────────────────────────────────
    # Weighted mean frequency — "brightness" of the sound
    centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)
    features.append(float(np.mean(centroid)))          # 1 value

    # ── Spectral rolloff ───────────────────────────────────────────────────
    # Frequency at which 85% of total energy is contained
    rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr, roll_percent=0.85)
    features.append(float(np.mean(rolloff)))           # 1 value

    # ── Spectral bandwidth ─────────────────────────────────────────────────
    # Standard deviation of the frequency distribution
    bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr)
    features.append(float(np.mean(bandwidth)))         # 1 value

    # ── Spectral contrast ──────────────────────────────────────────────────
    # Difference between peaks and valleys in 7 sub-bands
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
    features.extend(np.mean(contrast, axis=1))         # 7 values

    # ── Zero-crossing rate ─────────────────────────────────────────────────
    # How frequently the signal crosses zero — proxy for noisiness / percussiveness
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    features.append(float(np.mean(zcr)))               # 1 value

    # ── RMS energy ─────────────────────────────────────────────────────────
    rms = librosa.feature.rms(S=stft)
    features.append(float(np.mean(rms)))               # 1 value

    # ── Tempo ──────────────────────────────────────────────────────────────
    # BPM estimated via autocorrelation of onset strength
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    features.append(float(tempo))                      # 1 value

    return np.array(features, dtype=np.float32)  # shape: (51,)


# ── Main loop ──────────────────────────────────────────────────────────────
rows = []

for genre in GENRES:
    genre_dir = os.path.join(DATA_DIR, genre)
    if not os.path.isdir(genre_dir):
        print(f"WARNING: directory not found: {genre_dir}")
        continue

    files = [f for f in os.listdir(genre_dir) if f.endswith(".wav")]
    print(f"\nProcessing {genre} ({len(files)} files)...")

    for fname in tqdm(files, desc=genre):
        fpath = os.path.join(genre_dir, fname)
        feats = extract_features(fpath)
        if feats is not None:
            row = {"file": fname, "genre": genre}
            for i, v in enumerate(feats):
                row[f"f{i:03d}"] = float(v)
            rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(df)} rows to {OUTPUT_CSV}")
print(f"Feature matrix shape: {df.shape}")
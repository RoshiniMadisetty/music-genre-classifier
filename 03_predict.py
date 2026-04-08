import sys
import numpy as np
import joblib
import librosa

# ── Re-use the feature extractor from step 1 ──────────────────────────────
# (copy the extract_features function here or import it)
SR = 22050
N_MFCC = 13
HOP_LENGTH = 512
N_FFT = 2048

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SR, duration=30.0, mono=True)
    features = []
    stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc_delta = librosa.feature.delta(mfcc)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.mean(mfcc_delta, axis=1))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr, n_fft=N_FFT,
                                          hop_length=HOP_LENGTH)
    features.extend(np.mean(chroma, axis=1))
    centroid  = librosa.feature.spectral_centroid(S=stft, sr=sr)
    rolloff   = librosa.feature.spectral_rolloff(S=stft, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr)
    contrast  = librosa.feature.spectral_contrast(S=stft, sr=sr)
    zcr       = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    rms       = librosa.feature.rms(S=stft)
    tempo, _  = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    features.append(float(np.mean(centroid)))
    features.append(float(np.mean(rolloff)))
    features.append(float(np.mean(bandwidth)))
    features.extend(np.mean(contrast, axis=1))
    features.append(float(np.mean(zcr)))
    features.append(float(np.mean(rms)))
    features.append(float(tempo))
    return np.array(features, dtype=np.float32)

# ── Load saved pipeline ────────────────────────────────────────────────────
scaler = joblib.load("scaler.pkl")
pca    = joblib.load("pca.pkl")
svm    = joblib.load("svm_model.pkl")
le     = joblib.load("label_encoder.pkl")

# ── Predict ────────────────────────────────────────────────────────────────
audio_file = sys.argv[1] if len(sys.argv) > 1 else "test_song.wav"
print(f"Analysing: {audio_file}")

feats = extract_features(audio_file)          # (51,)
feats_scaled = scaler.transform([feats])       # (1, 51) — standardize
feats_pca    = pca.transform(feats_scaled)     # (1, k)  — project to PCs

# SVM decision function gives raw score per class
scores = svm.decision_function(feats_pca)[0]
probs  = np.exp(scores) / np.exp(scores).sum()  # softmax approximation

predicted_idx   = np.argmax(probs)
predicted_genre = le.classes_[predicted_idx]

print(f"\nPredicted genre: {predicted_genre.upper()}")
print("\nAll genre scores:")
for genre, prob in sorted(zip(le.classes_, probs),
                           key=lambda x: x[1], reverse=True):
    bar = "█" * int(prob * 40)
    print(f"  {genre:<12} {prob*100:5.1f}%  {bar}")
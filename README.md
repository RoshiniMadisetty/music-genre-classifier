# Music Genre Classification using PCA & Machine Learning

## 📌 Overview

This project classifies music into genres using audio signal processing and machine learning.

It combines:

* Feature extraction (MFCC)
* Linear algebra (PCA, eigenvalues)
* Machine learning (SVM, KNN)

---

## ⚙️ How it Works

1. Convert audio → numerical features (MFCC)
2. Apply PCA (eigenvectors) to reduce dimensions
3. Train classifiers (SVM, KNN)
4. Predict genre of new audio files

---

## 📂 Project Structure

```
music_genre_classifier/
│── data/                # Dataset (not included)
│── 01_extract_features.py
│── 02_train_model.py
│── 03_predict.py
│── 04_visualize_pca.py
│── requirements.txt
│── README.md
```

---

## 🚀 How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Add dataset

Download GTZAN dataset and place inside:

```
data/genres_original/
```

### 3. Run pipeline

```
python 01_extract_features.py
python 02_train_model.py
```

### 4. Predict a song

```
python 03_predict.py <audio_file.wav>
```

---

## 📊 Results

* Accuracy: ~75–80%
* PCA reduces features while preserving ~95% variance
* Classical and metal are easily separable
* Rock, pop, and country overlap

---

## 🧠 Key Concepts

* MFCC (Mel Frequency Cepstral Coefficients)
* Covariance Matrix
* Eigenvalues & Eigenvectors
* Principal Component Analysis (PCA)
* Support Vector Machines (SVM)

---

## 📌 Note

Dataset is not included due to size. Download from:
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

---



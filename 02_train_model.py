import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

# ── Load features ──────────────────────────────────────────────────────────
df = pd.read_csv("features.csv")
print(f"Dataset: {df.shape[0]} samples × {df.shape[1]-2} features")

# Remove non-numeric columns explicitly
X = df.drop(columns=["file", "genre"]).values
y_labels = df["genre"].values

# Encode genre strings → integers
le = LabelEncoder()
y = le.fit_transform(y_labels)       # e.g. blues=0, classical=1, ...
print(f"Classes: {list(le.classes_)}")

# ── Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── Step A: Standardize (critical before PCA) ──────────────────────────────
# Features have different scales (MFCC values ~[-10,10], tempo ~[60,180])
# StandardScaler makes each feature: mean=0, std=1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)          # use TRAIN stats only

# ── Step B: PCA — the linear algebra core ─────────────────────────────────
#
# Internally, sklearn PCA does:
#   1. Compute covariance matrix C = (1/n) * X^T X   shape: (51, 51)
#   2. Eigendecomposition: C * v = λ * v
#      - eigenvalues  λ₁ ≥ λ₂ ≥ ... ≥ λ₅₁  (variance explained per component)
#      - eigenvectors v₁, v₂, ..., v₅₁     (principal directions)
#   3. Project onto top-k eigenvectors:  X_pca = X * [v₁|v₂|...|vₖ]
#
# We choose k = number of components that explain 95% variance
pca = PCA(n_components=0.95, svd_solver="full", random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

n_comp = pca.n_components_
print(f"\nPCA: {X_train_scaled.shape[1]} features → {n_comp} components")
print(f"Variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# Print top-5 eigenvalues
ev = pca.explained_variance_
print("\nTop-10 eigenvalues (variance per principal component):")
for i, val in enumerate(ev[:10]):
    bar = "█" * int(val / ev[0] * 30)
    print(f"  PC{i+1:02d}: {val:7.2f}  {bar}")

# ── Step C: Classification ─────────────────────────────────────────────────
print("\n" + "="*55)
print("CLASSIFIER 1: K-Nearest Neighbours (k=5)")
print("="*55)
knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1)
knn.fit(X_train_pca, y_train)
y_pred_knn = knn.predict(X_test_pca)
acc_knn = accuracy_score(y_test, y_pred_knn)
print(f"Test accuracy: {acc_knn*100:.1f}%")
print(classification_report(y_test, y_pred_knn,
                             target_names=le.classes_, digits=3))

print("\n" + "="*55)
print("CLASSIFIER 2: Support Vector Machine (RBF kernel)")
print("="*55)
svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
svm.fit(X_train_pca, y_train)
y_pred_svm = svm.predict(X_test_pca)
acc_svm = accuracy_score(y_test, y_pred_svm)
print(f"Test accuracy: {acc_svm*100:.1f}%")
print(classification_report(y_test, y_pred_svm,
                             target_names=le.classes_, digits=3))

# Cross-validation on the best model
print("\n5-fold cross-validation (SVM):")
cv_scores = cross_val_score(
    SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
    pca.transform(scaler.transform(X)), y, cv=5, scoring="accuracy", n_jobs=-1
)
print(f"  Scores: {cv_scores.round(3)}")
print(f"  Mean ± Std: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

# ── Save model artifacts ───────────────────────────────────────────────────
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca,    "pca.pkl")
joblib.dump(svm,    "svm_model.pkl")
joblib.dump(le,     "label_encoder.pkl")
print("\nSaved: scaler.pkl, pca.pkl, svm_model.pkl, label_encoder.pkl")

# ── Confusion matrix plot ──────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred_svm)
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
ax.set_title(f"Confusion matrix — SVM (accuracy: {acc_svm*100:.1f}%)")
ax.set_ylabel("True genre")
ax.set_xlabel("Predicted genre")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Saved confusion_matrix.png")
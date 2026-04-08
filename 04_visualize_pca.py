import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# ── Load data ─────────────────────────────────────────────────────────────
df = pd.read_csv("features.csv")

# ✅ FIX: remove non-numeric columns explicitly
X = df.drop(columns=["file", "genre"]).values
y_labels = df["genre"].values

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y_labels)

# ── Scale features ────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── PCA (2D visualization) ────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)

ev = pca.explained_variance_ratio_

# ── Plot PCA scatter ──────────────────────────────────────────────────────
colors = plt.cm.tab10(np.linspace(0, 1, len(le.classes_)))

fig, ax = plt.subplots(figsize=(10, 8))

for i, genre in enumerate(le.classes_):
    mask = y == i
    ax.scatter(
        X_2d[mask, 0],
        X_2d[mask, 1],
        c=[colors[i]],
        s=20,
        alpha=0.6,
        label=genre
    )

ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% variance)")
ax.set_title("PCA of audio features — 2D projection")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pca_scatter.png", dpi=150)
plt.close()

print("Saved pca_scatter.png")

# ── Scree plot (eigenvalues) ─────────────────────────────────────────────
pca_full = PCA(random_state=42)
pca_full.fit(X_scaled)

ev_full = pca_full.explained_variance_ratio_
cumulative = np.cumsum(ev_full) * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Scree (bar)
ax1.bar(range(1, 21), ev_full[:20] * 100)
ax1.set_xlabel("Principal component")
ax1.set_ylabel("Variance explained (%)")
ax1.set_title("Scree plot — top 20 PCs")

# Cumulative
ax2.plot(range(1, len(ev_full)+1), cumulative)
ax2.axhline(95, linestyle="--")
k95 = np.argmax(cumulative >= 95) + 1
ax2.axvline(k95, linestyle="--")

ax2.set_xlabel("Number of components")
ax2.set_ylabel("Cumulative variance (%)")
ax2.set_title("Cumulative explained variance")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("scree_plot.png", dpi=150)
plt.close()

print(f"Saved scree_plot.png")
print(f"Components needed for 95% variance: {k95}")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("features.csv")
feature_cols = [c for c in df.columns if c.startswith("f")]
X = df[feature_cols].values
le = LabelEncoder()
y = le.fit_transform(df["genre"].values)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)

ev = pca.explained_variance_ratio_
colors = plt.cm.tab10(np.linspace(0, 1, 10))

fig, ax = plt.subplots(figsize=(10, 8))
for i, genre in enumerate(le.classes_):
    mask = y == i
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=[colors[i]], s=20, alpha=0.6, label=genre)

ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% variance)")
ax.set_title("PCA of audio features — 2D projection (1000 songs, 10 genres)")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pca_scatter.png", dpi=150)
plt.show()

# ── Scree plot (eigenvalues) ───────────────────────────────────────────────
pca_full = PCA(random_state=42)
pca_full.fit(X_scaled)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ev_full = pca_full.explained_variance_ratio_
ax1.bar(range(1, 21), ev_full[:20] * 100, color="steelblue", alpha=0.8)
ax1.set_xlabel("Principal component")
ax1.set_ylabel("Variance explained (%)")
ax1.set_title("Scree plot — eigenvalues (top 20 PCs)")

cumulative = np.cumsum(ev_full) * 100
ax2.plot(range(1, len(ev_full)+1), cumulative, "b-", linewidth=2)
ax2.axhline(95, color="red", linestyle="--", label="95% threshold")
k95 = np.argmax(cumulative >= 95) + 1
ax2.axvline(k95, color="orange", linestyle="--", label=f"k={k95}")
ax2.set_xlabel("Number of components")
ax2.set_ylabel("Cumulative variance (%)")
ax2.set_title("Cumulative explained variance")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("scree_plot.png", dpi=150)
plt.show()
print(f"Components needed for 95% variance: {k95}")
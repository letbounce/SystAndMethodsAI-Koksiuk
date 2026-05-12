"""
ЛР-7, завдання 2.3: оцінка кількості кластерів методом зсуву середнього (Mean Shift)
на даних data_clustering.txt.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data_clustering.txt")
OUT_DIR = os.path.join(BASE_DIR, "outputs_task3")
os.makedirs(OUT_DIR, exist_ok=True)

X = np.loadtxt(DATA_PATH, delimiter=",")

# Оцінка ширини вікна (bandwidth); quantile більший → ширше вікно → менше кластерів
bandwidth = estimate_bandwidth(X, quantile=0.15, n_samples=min(500, len(X)))

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
n_clusters = len(np.unique(labels))

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap="tab10", edgecolors="k", linewidths=0.3)
plt.scatter(
    cluster_centers[:, 0],
    cluster_centers[:, 1],
    c="red",
    s=250,
    marker="*",
    edgecolors="black",
    label="Центри Mean Shift",
)
plt.title(f"Mean Shift: знайдено кластерів = {n_clusters}, bandwidth={bandwidth:.4f}")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_mean_shift_clusters.png"), dpi=160)
plt.close()

print("Estimated bandwidth:", bandwidth)
print("Number of clusters:", n_clusters)
print("Cluster centers:\n", cluster_centers)
print(f"Plot saved to: {OUT_DIR}")

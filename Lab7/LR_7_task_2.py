"""
ЛР-7, завдання 2.2: K-means для набору Iris (виправлений код на основі підказки методички).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances_argmin

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs_task2")
os.makedirs(OUT_DIR, exist_ok=True)

# Завантаження Iris
iris = load_iris()
X = iris["data"]
y_true = iris["target"]

# KMeans з k=3 (три види ірису), k-means++
kmeans = KMeans(
    n_clusters=3,
    init="k-means++",
    n_init=10,
    max_iter=300,
    random_state=0,
)
y_kmeans = kmeans.fit_predict(X)

# Візуалізація у проекції перші дві ознаки (довжина/ширина чашолистка)
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=45, cmap="viridis", edgecolors="k", linewidths=0.3)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c="red", s=200, marker="X", edgecolors="black", label="Центроїди")
plt.title("Iris: K-means (k=3), проекція на ознаки 0 та 1")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_iris_kmeans_features_0_1.png"), dpi=160)
plt.close()


def find_clusters(X_data: np.ndarray, n_clusters: int, rseed: int = 2):
    """Спрощена ручна імітація k-means (як у методичці), для порівняння."""
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X_data.shape[0])[:n_clusters]
    centers_local = X_data[i].copy()
    while True:
        labels_local = pairwise_distances_argmin(X_data, centers_local)
        new_centers = np.array([X_data[labels_local == j].mean(0) for j in range(n_clusters)])
        if np.allclose(centers_local, new_centers):
            break
        centers_local = new_centers
    return centers_local, labels_local


fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)

centers_a, labels_a = find_clusters(X[:, :2], 3, rseed=2)
axes[0].scatter(X[:, 0], X[:, 1], c=labels_a, s=40, cmap="viridis", edgecolors="k", linewidths=0.2)
axes[0].set_title("find_clusters, rseed=2")

centers_b, labels_b = find_clusters(X[:, :2], 3, rseed=0)
axes[1].scatter(X[:, 0], X[:, 1], c=labels_b, s=40, cmap="viridis", edgecolors="k", linewidths=0.2)
axes[1].set_title("find_clusters, rseed=0")

labels_sk = KMeans(3, random_state=0, n_init=10, init="k-means++").fit_predict(X[:, :2])
axes[2].scatter(X[:, 0], X[:, 1], c=labels_sk, s=40, cmap="viridis", edgecolors="k", linewidths=0.2)
axes[2].set_title("sklearn KMeans, random_state=0")

for ax in axes:
    ax.set_xlabel(iris.feature_names[0])
    ax.set_ylabel(iris.feature_names[1])
    ax.grid(alpha=0.25)
plt.suptitle("Порівнення ініціалізації (лише перші 2 ознаки)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_iris_init_comparison.png"), dpi=160)
plt.close()

# Adjusted Rand Index (опційно) — оцінка узгодженості з істинними мітками
from sklearn.metrics import adjusted_rand_score

ari = adjusted_rand_score(y_true, y_kmeans)
print("Adjusted Rand Index (KMeans labels vs true species):", round(ari, 4))
print(f"Plots saved to: {OUT_DIR}")

"""
ЛР-7, завдання 2.1: кластеризація методом k-середніх (data_clustering.txt).
Зберігає графіки у outputs_task1/ для включення у звіт.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Завантаження вхідних даних
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data_clustering.txt")
OUT_DIR = os.path.join(BASE_DIR, "outputs_task1")
os.makedirs(OUT_DIR, exist_ok=True)

X = np.loadtxt(DATA_PATH, delimiter=",")

# Візуалізація вхідних даних
plt.figure(figsize=(7, 6))
plt.scatter(X[:, 0], X[:, 1], s=30, edgecolors="k", facecolors="none")
plt.title("Вхідні дані (data_clustering.txt)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_input_scatter.png"), dpi=160)
plt.close()

# Створення об'єкта KMeans (k-means++, 5 кластерів — як у методичці)
kmeans = KMeans(
    n_clusters=5,
    init="k-means++",
    n_init=10,
    max_iter=300,
    tol=1e-4,
    random_state=7,
)
kmeans.fit(X)

# Визначення кроку сітки та області для візуалізації меж
h = 0.05
x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Передбачення міток для усіх точок сітки
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.35, cmap=plt.cm.Set3)
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=35, edgecolors="k", cmap="viridis")
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c="red",
    s=220,
    marker="X",
    edgecolors="black",
    linewidths=1.2,
    label="Центроїди",
)
plt.title("K-means (k=5): області кластерів та центроїди")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_kmeans_regions_and_centers.png"), dpi=160)
plt.close()

# Інерція (сума квадратів відстаней до центроїдів) — для звіту
print("Inertia (within-cluster sum of squares):", round(float(kmeans.inertia_), 4))
print("Cluster sizes:", np.bincount(kmeans.labels_, minlength=5).tolist())
print("Centroids:\n", kmeans.cluster_centers_)
print(f"Plots saved to: {OUT_DIR}")

"""
ЛР-7, завдання 2.4: підгрупи на фондових даних (matplotlib sample_data) —
Affinity Propagation.

У сучасних версіях matplotlib файл company_symbol_mapping.json може бути відсутній;
використовується локальний company_symbol_mapping.json у папці Lab7.
Дані: Stocks.csv (місячні котирування). Оскільки у файлі немає окремих Open/Close,
як ознаку «варіації» беремо приріст ціни між сусідніми датами: Δp_t = p_t - p_{t-1}.
"""

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs_task4")
os.makedirs(OUT_DIR, exist_ok=True)

# --- Прив'язка символів до назв ---
mapping_path = os.path.join(BASE_DIR, "company_symbol_mapping.json")
with open(mapping_path, "r", encoding="utf-8") as f:
    symbol_to_name = json.load(f)

# --- Завантаження Stocks.csv з matplotlib ---
sample_dir = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "sample_data")
stocks_path = os.path.join(sample_dir, "Stocks.csv")

df = pd.read_csv(stocks_path, skiprows=1)
df = df.rename(columns={df.columns[0]: "Date"})
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

# Числові колонки — котирування по тикерах
tickers = [c for c in df.columns if c in symbol_to_name]
prices = df[tickers].apply(pd.to_numeric, errors="coerce")

# Варіація між послідовними спостереженнями (аналог зміни між «сеансами»)
returns = prices.diff().dropna(how="all")
returns = returns.dropna(axis=0, how="all")

# Останні 48 місяців як вектор ознак для кожної компанії/індексу
window = 48
tail = returns.iloc[-window:]
feature_matrix = tail.T.values  # рядки = тикери, стовпці = час

# Прибрати рядки з надто багатьма NaN
valid_mask = np.isfinite(feature_matrix).sum(axis=1) >= window // 2
symbols_valid = np.array(tickers)[valid_mask]
X = feature_matrix[valid_mask]
X = np.nan_to_num(X, nan=0.0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# AffinityPropagation: preference контролює кількість екземплярів
# Підбираємо preference як медіану від'ємних квадратів відстаней (типова евристика)
S = -euclidean_distances(X_scaled, squared=True)
preference = float(np.median(S)) * 0.85

ap = AffinityPropagation(
    affinity="precomputed",
    damping=0.9,
    max_iter=500,
    convergence_iter=30,
    preference=preference,
    random_state=7,
)
labels = ap.fit_predict(S)
n_clusters = len(set(labels))
exemplars_idx = ap.cluster_centers_indices_

# 2D-проекція лише для візуалізації (ознак багато — PCA зберігає структуру для графіка)
pca = PCA(n_components=2, random_state=7)
X_2d = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=labels,
    s=120,
    cmap="tab10",
    edgecolors="black",
)
for i, sym in enumerate(symbols_valid):
    plt.annotate(sym, (X_2d[i, 0], X_2d[i, 1]), fontsize=8, xytext=(4, 4), textcoords="offset points")
ex_2d = X_2d[exemplars_idx]
plt.scatter(
    ex_2d[:, 0],
    ex_2d[:, 1],
    s=260,
    facecolors="none",
    edgecolors="red",
    linewidths=2,
    label="Exemplars (зразки)",
)
plt.title(f"Affinity Propagation на варіаціях котирувань: кластерів = {n_clusters}")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_affinity_propagation_2d_projection.png"), dpi=160)
plt.close()

# Текстовий звіт
report_lines = [
    f"Symbols used ({len(symbols_valid)}): {', '.join(symbols_valid.tolist())}",
    f"Number of clusters: {n_clusters}",
    "Exemplar per cluster (index in X -> ticker):",
]
for cluster_id, ex_row in enumerate(exemplars_idx):
    ex_sym = symbols_valid[ex_row]
    members = symbols_valid[labels == cluster_id].tolist()
    report_lines.append(f"  cluster {cluster_id}: exemplar={ex_sym} (row {ex_row}), members={members}")

report_path = os.path.join(OUT_DIR, "affinity_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("\n".join(report_lines))
print(f"\nSaved plot: {os.path.join(OUT_DIR, '01_affinity_propagation_2d_projection.png')}")
print(f"Saved report: {report_path}")

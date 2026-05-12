"""
ЛР-8 (методичка «TensorFlow / навчання лінійної регресії»).

Реалізація алгоритму з методички:
  y = 2*x + 1 + ε,  ε ~ N(0, 2)   (дисперсія шуму = 2 → std = sqrt(2))
  модель: ŷ = k*x + b, параметри k, b навчаються мінімізацією
  loss = mean_i (y_i - ŷ_i)^2 на міні-батчі (у методичці часто reduce_sum;
  тут reduce_mean для стабільніших градієнтів SGD — еквівалент до іншого масштабу кроку).

У TensorFlow 2 використовується tf.compat.v1 (граф + Session), щоб зберегти
структуру «placeholder → змінні → loss → optimizer → sess.run».
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Графовий режим TF1 API у TF2
tf.compat.v1.disable_eager_execution()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# --- 1) Дані ---
rng = np.random.default_rng(7)
n_samples = 1000
X_data = rng.uniform(0.0, 1.0, size=(n_samples, 1)).astype(np.float32)
noise = rng.normal(loc=0.0, scale=np.sqrt(2.0), size=(n_samples, 1)).astype(np.float32)
y_data = (2.0 * X_data + 1.0 + noise).astype(np.float32).reshape(-1)

# --- 2) Граф TensorFlow (compat.v1) ---
graph = tf.Graph()
with graph.as_default():
    X = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name="X")
    y = tf.compat.v1.placeholder(tf.float32, shape=[None], name="y")

    # 3) Параметри k, b
    # Менш «агресивна» ініціалізація, ніж N(0,1), щоб уникнути вибуху градієнтів на першому кроці
    k = tf.compat.v1.get_variable("k", shape=(), initializer=tf.zeros_initializer())
    b = tf.compat.v1.get_variable("b", shape=(), initializer=tf.zeros_initializer())

    y_hat = tf.squeeze(k * X + b, axis=[-1])

    # Методичка згадує reduce_sum по батчу; для стабільної SGD у TF2 зручніше
    # мінімізувати середній квадрат помилки (еквівалентно до постійного масштабу 1/N_batch).
    loss = tf.reduce_mean(tf.square(y - y_hat))

    # 5) Оптимізатор (стохастичний градієнтний спуск у методичці)
    # При середньому MSE по батчу можна використовувати більший крок, ніж для reduce_sum
    learning_rate = 0.08
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    init = tf.compat.v1.global_variables_initializer()

# --- 6) Навчальний цикл ---
n_epochs = 20_000
batch_size = 100
log_every = 100

loss_hist = []
k_hist = []
b_hist = []

with tf.compat.v1.Session(graph=graph) as sess:
    sess.run(init)
    for epoch in range(1, n_epochs + 1):
        idx = rng.choice(n_samples, size=batch_size, replace=False)
        feed = {X: X_data[idx], y: y_data[idx]}
        _, loss_val, k_val, b_val = sess.run([optimizer, loss, k, b], feed_dict=feed)
        loss_hist.append(float(loss_val))
        k_hist.append(float(k_val))
        b_hist.append(float(b_val))
        if epoch % log_every == 0 or epoch == 1:
            print(f"Епоха {epoch:5d}: loss={loss_val:.6f}, k={k_val:.4f}, b={b_val:.4f}")

    k_final, b_final = sess.run([k, b])

print("\nОцінки після навчання:")
print(f"  k ~ {k_final:.6f}  (істинне 2.0)")
print(f"  b ~ {b_final:.6f}  (істинне 1.0)")

# --- Графіки ---
plt.figure(figsize=(8, 5))
plt.plot(loss_hist, lw=0.8)
plt.xlabel("Ітерація")
plt.ylabel("Loss (MSE у батчі)")
plt.title("Динаміка функції втрат під час SGD")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_loss_curve.png"), dpi=160)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(k_hist, label="k")
plt.plot(b_hist, label="b")
plt.axhline(2.0, color="tab:blue", ls="--", lw=1, alpha=0.7, label="істинне k=2")
plt.axhline(1.0, color="tab:orange", ls="--", lw=1, alpha=0.7, label="істинне b=1")
plt.xlabel("Ітерація")
plt.ylabel("Значення параметра")
plt.title("Збіжність параметрів k та b")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_kb_convergence.png"), dpi=160)
plt.close()

plt.figure(figsize=(7, 6))
plt.scatter(X_data.reshape(-1), y_data, s=8, alpha=0.35, label="дані")
xs = np.linspace(0, 1, 200, dtype=np.float32)
plt.plot(xs, k_final * xs + b_final, "r-", lw=2, label=f"навчена пряма: y={k_final:.3f}x+{b_final:.3f}")
plt.plot(xs, 2.0 * xs + 1.0, "k--", lw=1.5, alpha=0.8, label="істинна: y=2x+1")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(alpha=0.3)
plt.title("Лінійна регресія: дані та пряма")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_data_and_fit.png"), dpi=160)
plt.close()

metrics_path = os.path.join(OUT_DIR, "training_summary.txt")
with open(metrics_path, "w", encoding="utf-8") as f:
    f.write(f"final_k={k_final}\nfinal_b={b_final}\nfinal_batch_loss={loss_hist[-1]}\n")
print(f"\nЗбережено графіки та {metrics_path}")

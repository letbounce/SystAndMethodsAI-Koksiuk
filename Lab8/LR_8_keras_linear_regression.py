"""
Додатково до методички (сучасний Keras 3 / TensorFlow): та сама лінійна регресія
через `keras.Sequential` + Dense(1, use_bias=True).

Це ілюструє розділ методички про моделі Keras поруч із «нижньорівневим» TF graph API.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.layers import Dense, Input
from keras.optimizers import SGD

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs_keras")
os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(11)
n = 1000
X = rng.uniform(0, 1, size=(n, 1)).astype(np.float32)
y = (2.0 * X + 1.0 + rng.normal(0, np.sqrt(2.0), size=(n, 1))).astype(np.float32)

model = Sequential(
    [
        Input(shape=(1,)),
        Dense(1, use_bias=True, kernel_initializer="zeros", bias_initializer="zeros"),
    ]
)
model.compile(optimizer=SGD(learning_rate=0.2), loss="mse")

history = model.fit(X, y, epochs=400, batch_size=100, verbose=0)

k_hat = float(model.layers[0].get_weights()[0].reshape(-1)[0])
b_hat = float(model.layers[0].get_weights()[1].reshape(-1)[0])
print(f"Keras Dense: k~{k_hat:.6f}, b~{b_hat:.6f}")

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"])
plt.title("Keras: MSE під час навчання")
plt.xlabel("Епоха")
plt.ylabel("MSE")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "keras_mse.png"), dpi=160)
plt.close()

xs = np.linspace(0, 1, 200, dtype=np.float32).reshape(-1, 1)
plt.figure(figsize=(7, 6))
plt.scatter(X, y, s=8, alpha=0.35)
plt.plot(xs, model.predict(xs, verbose=0), "r-", lw=2, label="Keras прогноз")
plt.plot(xs, 2.0 * xs + 1.0, "k--", lw=1.5, label="y=2x+1")
plt.legend()
plt.grid(alpha=0.3)
plt.title("Keras: дані та навчена пряма")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "keras_fit.png"), dpi=160)
plt.close()

print(f"Saved plots to: {OUT_DIR}")

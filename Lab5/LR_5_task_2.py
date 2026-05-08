import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Task 2.2: Handle class imbalance with ExtraTrees"
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Enable class_weight='balanced' to handle class imbalance",
    )
    parser.add_argument(
        "--ignore",
        action="store_true",
        help="Use zero_division=0 for stable metric output",
    )
    return parser


def plot_binary_data(X: np.ndarray, y: np.ndarray, out_path: str, title: str) -> None:
    cls0 = X[y == 0]
    cls1 = X[y == 1]

    plt.figure(figsize=(7, 5))
    plt.scatter(cls0[:, 0], cls0[:, 1], s=40, marker="o", edgecolor="black", label="Class 0")
    plt.scatter(cls1[:, 0], cls1[:, 1], s=40, marker="s", edgecolor="black", label="Class 1")
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_boundary_binary(clf, X, y, out_path: str, title: str) -> None:
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300)
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, Z, alpha=0.25, cmap=plt.cm.Paired)

    for class_id, marker in zip([0, 1], ["o", "s"]):
        pts = X[y == class_id]
        plt.scatter(
            pts[:, 0],
            pts[:, 1],
            s=35,
            marker=marker,
            edgecolor="black",
            label=f"Class {class_id}",
        )

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(alpha=0.2)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    args = build_arg_parser().parse_args()
    balance = args.balance
    zero_division = 0 if args.ignore else "warn"

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "data_imbalance.txt")
    mode_name = "balanced" if balance else "unbalanced"
    out_dir = os.path.join(base_dir, "outputs_task2", mode_name)
    os.makedirs(out_dir, exist_ok=True)

    data = np.loadtxt(input_file, delimiter=",")
    X, y = data[:, :-1], data[:, -1].astype(int)

    plot_binary_data(X, y, os.path.join(out_dir, "input_data.png"), "Task 2.2 - Input data")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=5, stratify=y
    )

    params = {"n_estimators": 200, "max_depth": 8, "random_state": 7}
    if balance:
        params["class_weight"] = "balanced"

    clf = ExtraTreesClassifier(**params)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)

    plot_boundary_binary(
        clf,
        X_test,
        y_test,
        os.path.join(out_dir, "test_boundary.png"),
        f"Task 2.2 - Test boundary ({mode_name})",
    )

    print("\n" + "#" * 70)
    print(f"Task 2.2 mode: {mode_name}")
    print("#" * 70)
    print("\nTrain report:\n")
    print(
        classification_report(
            y_train, y_pred_train, target_names=["Class-0", "Class-1"], zero_division=zero_division
        )
    )
    print("\nTest report:\n")
    print(
        classification_report(
            y_test, y_pred, target_names=["Class-0", "Class-1"], zero_division=zero_division
        )
    )
    print("Balanced accuracy:", round(balanced_accuracy_score(y_test, y_pred), 4))
    print("Weighted F1 score:", round(f1_score(y_test, y_pred, average="weighted"), 4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"\nSaved plots to: {out_dir}")


if __name__ == "__main__":
    main()


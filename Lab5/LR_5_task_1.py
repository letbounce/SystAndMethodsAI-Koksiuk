import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Task 2.1: Random Forest / Extra Trees classification"
    )
    parser.add_argument(
        "--classifier-type",
        dest="classifier_type",
        required=False,
        default="rf",
        choices=["rf", "erf"],
        help="rf = RandomForestClassifier, erf = ExtraTreesClassifier",
    )
    return parser


def plot_input_data(X: np.ndarray, y: np.ndarray, out_path: str) -> None:
    class_0 = X[y == 0]
    class_1 = X[y == 1]
    class_2 = X[y == 2]

    plt.figure(figsize=(7, 5))
    plt.scatter(
        class_0[:, 0], class_0[:, 1], s=45, marker="s", edgecolor="black", label="Class 0"
    )
    plt.scatter(
        class_1[:, 0], class_1[:, 1], s=45, marker="o", edgecolor="black", label="Class 1"
    )
    plt.scatter(
        class_2[:, 0], class_2[:, 1], s=45, marker="^", edgecolor="black", label="Class 2"
    )
    plt.title("Task 2.1 - Input data")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_decision_regions(
    clf, X: np.ndarray, y: np.ndarray, title: str, out_path: str, overlay_points=None
) -> None:
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300)
    )

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, Z, alpha=0.28, cmap=plt.cm.Set3)

    for class_id, marker in zip([0, 1, 2], ["s", "o", "^"]):
        points = X[y == class_id]
        plt.scatter(
            points[:, 0],
            points[:, 1],
            s=35,
            marker=marker,
            edgecolor="black",
            label=f"Class {class_id}",
        )

    if overlay_points is not None:
        overlay_points = np.asarray(overlay_points)
        plt.scatter(
            overlay_points[:, 0],
            overlay_points[:, 1],
            c="red",
            s=80,
            marker="*",
            label="Test datapoints",
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
    classifier_type = args.classifier_type

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "data_random_forests.txt")
    out_dir = os.path.join(base_dir, "outputs_task1")
    os.makedirs(out_dir, exist_ok=True)

    data = np.loadtxt(input_file, delimiter=",")
    X, y = data[:, :-1], data[:, -1].astype(int)

    plot_input_data(X, y, os.path.join(out_dir, f"{classifier_type}_input_data.png"))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=5, stratify=y
    )

    params = {"n_estimators": 200, "max_depth": 6, "random_state": 7}
    if classifier_type == "rf":
        classifier = RandomForestClassifier(**params)
    else:
        classifier = ExtraTreesClassifier(**params)

    classifier.fit(X_train, y_train)

    plot_decision_regions(
        classifier,
        X_train,
        y_train,
        f"Training decision boundary ({classifier_type})",
        os.path.join(out_dir, f"{classifier_type}_train_boundary.png"),
    )
    plot_decision_regions(
        classifier,
        X_test,
        y_test,
        f"Test decision boundary ({classifier_type})",
        os.path.join(out_dir, f"{classifier_type}_test_boundary.png"),
    )

    y_test_pred = classifier.predict(X_test)
    class_names = ["Class-0", "Class-1", "Class-2"]
    print("\n" + "#" * 70)
    print(f"Classifier: {classifier.__class__.__name__} ({classifier_type})")
    print("#" * 70)
    print("\nTraining classification report:\n")
    print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
    print("\nTest classification report:\n")
    print(classification_report(y_test, y_test_pred, target_names=class_names))

    test_datapoints = np.array([[5, 5], [3, 6], [6, 4], [7, 2], [4, 4], [5, 2]], dtype=float)
    print("\nConfidence measure on custom datapoints:")
    for datapoint in test_datapoints:
        probabilities = classifier.predict_proba([datapoint])[0]
        predicted_class = int(np.argmax(probabilities))
        print(
            f"Datapoint={datapoint.tolist()} | predicted=Class-{predicted_class} | "
            f"proba={np.round(probabilities, 4).tolist()}"
        )

    plot_decision_regions(
        classifier,
        X,
        y,
        f"Custom test datapoints over boundary ({classifier_type})",
        os.path.join(out_dir, f"{classifier_type}_custom_points.png"),
        overlay_points=test_datapoints,
    )

    print(f"\nSaved plots to: {out_dir}")


if __name__ == "__main__":
    main()


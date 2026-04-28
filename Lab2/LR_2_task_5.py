import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def main():
    sns.set()

    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )

    clf = RidgeClassifier(tol=1e-2, solver="sag")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Accuracy:", np.round(metrics.accuracy_score(y_test, y_pred), 4))
    print("Precision (weighted):", np.round(metrics.precision_score(y_test, y_pred, average="weighted"), 4))
    print("Recall (weighted):", np.round(metrics.recall_score(y_test, y_pred, average="weighted"), 4))
    print("F1 Score (weighted):", np.round(metrics.f1_score(y_test, y_pred, average="weighted"), 4))
    print("Cohen Kappa Score:", np.round(metrics.cohen_kappa_score(y_test, y_pred), 4))
    print("Matthews Corrcoef:", np.round(metrics.matthews_corrcoef(y_test, y_pred), 4))
    print("\nClassification Report:\n", metrics.classification_report(y_test, y_pred, target_names=iris.target_names))

    mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        mat,
        square=True,
        annot=True,
        fmt="d",
        cbar=False,
        xticklabels=iris.target_names,
        yticklabels=iris.target_names,
    )
    plt.xlabel("predicted label")
    plt.ylabel("true label")
    plt.title("Confusion Matrix (RidgeClassifier on Iris)")
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Confusion.jpg")
    plt.savefig(out_path, dpi=200)
    plt.show()

    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()


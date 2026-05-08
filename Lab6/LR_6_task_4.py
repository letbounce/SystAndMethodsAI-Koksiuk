import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "outputs_task4")
    os.makedirs(out_dir, exist_ok=True)

    url = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv"
    df = pd.read_csv(url)

    # Minimal cleaning
    df = df.drop_duplicates().copy()
    df["insert_date"] = pd.to_datetime(df["insert_date"], errors="coerce")
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")

    # Derive numerical time features
    df["travel_minutes"] = (df["end_date"] - df["start_date"]).dt.total_seconds() / 60.0
    df["insert_hour"] = df["insert_date"].dt.hour
    df["departure_hour"] = df["start_date"].dt.hour
    df["departure_dayofweek"] = df["start_date"].dt.dayofweek

    # Target for Bayes classification:
    # We classify ticket price into 3 categories (cheap/medium/expensive) by quantiles.
    # This turns a real ticket-pricing dataset into a supervised classification problem.
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"]).copy()
    price = df["price"].astype(float)
    q1, q2 = price.quantile([0.33, 0.66]).to_list()
    df["price_class"] = pd.cut(
        price,
        bins=[-np.inf, q1, q2, np.inf],
        labels=["cheap", "medium", "expensive"],
    ).astype(str)

    feature_cols = [
        "origin",
        "destination",
        "train_type",
        "train_class",
        "fare",
        "travel_minutes",
        "insert_hour",
        "departure_hour",
        "departure_dayofweek",
    ]

    data = df[feature_cols + ["price_class"]].copy()
    X = data[feature_cols]
    y = data["price_class"]

    categorical_cols = ["origin", "destination", "train_type", "train_class", "fare"]
    numeric_cols = ["travel_minutes", "insert_hour", "departure_hour", "departure_dayofweek"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    # Dense output required for GaussianNB
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7, stratify=y
    )

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Convert sparse matrices to dense if needed
    if hasattr(X_train_t, "toarray"):
        X_train_t = X_train_t.toarray()
    if hasattr(X_test_t, "toarray"):
        X_test_t = X_test_t.toarray()

    clf = GaussianNB()
    clf.fit(X_train_t, y_train)
    y_pred = clf.predict(X_test_t)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=["cheap", "medium", "expensive"])

    print("\n" + "#" * 78)
    print("Lab6 Task 4 - Naive Bayes analysis on Renfe ticket prices")
    print("#" * 78)
    print(f"Rows used: {len(data)}")
    print(f"Price quantiles for classes: q33={q1:.2f}, q66={q2:.2f}")
    print(f"Test accuracy: {acc:.4f}")
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix (labels: cheap, medium, expensive):\n", cm)

    # Save confusion matrix figure
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Task 4 - Confusion matrix (GaussianNB)")
    plt.colorbar()
    tick_labels = ["cheap", "medium", "expensive"]
    plt.xticks(range(3), tick_labels, rotation=25)
    plt.yticks(range(3), tick_labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "task4_confusion_matrix.png")
    plt.savefig(cm_path, dpi=180)
    plt.close()

    # Save top-level metrics summary
    metrics_path = os.path.join(out_dir, "task4_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"rows={len(data)}\n")
        f.write(f"q33={q1:.4f}, q66={q2:.4f}\n")
        f.write(f"accuracy={acc:.6f}\n")
        f.write(f"macro_precision={report['macro avg']['precision']:.6f}\n")
        f.write(f"macro_recall={report['macro avg']['recall']:.6f}\n")
        f.write(f"macro_f1={report['macro avg']['f1-score']:.6f}\n")
        f.write("confusion_matrix=\n")
        f.write(np.array2string(cm))

    print(f"\nSaved files:\n- {cm_path}\n- {metrics_path}")


if __name__ == "__main__":
    main()


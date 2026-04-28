import os

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def load_income_data(path: str) -> pd.DataFrame:
    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]
    df = pd.read_csv(
        path,
        header=None,
        names=columns,
        sep=r",\s*",
        engine="python",
        na_values="?",
    )
    return df.dropna().copy()


def main():
    lab2_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(lab2_dir, "income_data.txt")

    df = load_income_data(input_path)
    max_per_class = 25_000
    df_le = df[df["income"] == "<=50K"].head(max_per_class)
    df_gt = df[df["income"] == ">50K"].head(max_per_class)
    df_balanced = (
        pd.concat([df_le, df_gt], axis=0)
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )

    X_df = df_balanced.drop(columns=["income"])
    y = df_balanced["income"]

    numeric_features = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    categorical_features = [c for c in X_df.columns if c not in numeric_features]

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocess_dense = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", ohe),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    models = []
    models.append(("LR", LogisticRegression(solver="liblinear", max_iter=200)))
    models.append(("LDA", LinearDiscriminantAnalysis()))
    models.append(("KNN", KNeighborsClassifier(n_neighbors=15)))
    models.append(("CART", DecisionTreeClassifier(random_state=1)))
    models.append(("NB", GaussianNB()))
    models.append(("SVM", SVC(gamma="scale")))

    kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

    results = []
    names_ = []
    for name, clf in models:
        pipe = Pipeline(steps=[("preprocess", preprocess_dense), ("clf", clf)])
        scores = cross_val_score(pipe, X_df, y, cv=kfold, scoring="accuracy")
        results.append(scores)
        names_.append(name)
        print(f"{name}: {scores.mean():.4f} ({scores.std():.4f})")

    plt.figure(figsize=(8, 4))
    plt.boxplot(results, labels=names_)
    plt.title("Algorithm Comparison (income_data.txt)")
    plt.ylabel("Accuracy (CV)")
    plt.show()

    # Optional: simple hold-out split (same as in notebook)
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=5, stratify=y
    )
    best = Pipeline(
        steps=[("preprocess", preprocess_dense), ("clf", LogisticRegression(solver="liblinear", max_iter=200))]
    )
    best.fit(X_train, y_train)
    print("\nExample model trained. You can pick the best by CV results.")


if __name__ == "__main__":
    main()


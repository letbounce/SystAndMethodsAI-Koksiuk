import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "data_random_forests.txt")

    data = np.loadtxt(input_file, delimiter=",")
    X, y = data[:, :-1], data[:, -1].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=5, stratify=y
    )

    param_grid = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 4, 6, 8, None],
        "min_samples_split": [2, 4, 8],
    }

    print("\n" + "#" * 70)
    print("Task 2.3 - Grid search for best parameters")
    print("#" * 70)

    for scoring in ["precision_macro", "recall_macro"]:
        print(f"\nScoring metric: {scoring}")
        grid = GridSearchCV(
            RandomForestClassifier(random_state=7),
            param_grid=param_grid,
            scoring=scoring,
            cv=5,
            n_jobs=-1,
            verbose=0,
            return_train_score=True,
        )
        grid.fit(X_train, y_train)

        print("Best params:", grid.best_params_)
        print("Best CV score:", round(grid.best_score_, 4))

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        print("\nTest classification report:")
        print(classification_report(y_test, y_pred, target_names=["Class-0", "Class-1", "Class-2"]))

        # Top 10 combinations for transparency
        results = grid.cv_results_
        order = np.argsort(results["rank_test_score"])[:10]
        print("Top parameter combinations:")
        for idx in order:
            print(
                f"rank={results['rank_test_score'][idx]:2d} | "
                f"mean={results['mean_test_score'][idx]:.4f} | "
                f"std={results['std_test_score'][idx]:.4f} | "
                f"params={results['params'][idx]}"
            )


if __name__ == "__main__":
    main()


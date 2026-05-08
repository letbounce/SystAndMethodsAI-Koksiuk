import os

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "traffic_data.txt")

    # day_of_week, time_of_day, opponent_team, game_on(yes/no), traffic_count
    raw_data = np.loadtxt(input_file, delimiter=",", dtype=str)

    X_raw = raw_data[:, :-1]
    y = raw_data[:, -1].astype(float)

    X_encoded = np.empty(X_raw.shape, dtype=float)
    label_encoders: dict[int, LabelEncoder] = {}

    # Encode all 4 input columns (they are categorical strings in this dataset).
    for col in range(X_raw.shape[1]):
        enc = LabelEncoder()
        X_encoded[:, col] = enc.fit_transform(X_raw[:, col])
        label_encoders[col] = enc

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.25, random_state=7
    )

    regressor = ExtraTreesRegressor(
        n_estimators=400,
        max_depth=None,
        random_state=7,
        n_jobs=-1,
    )
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n" + "#" * 70)
    print("Task 2.5 - Traffic intensity prediction (ExtraTreesRegressor)")
    print("#" * 70)
    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"R2   = {r2:.4f}")

    # Test unknown sample in original string format:
    unknown_datapoint = ["Tuesday", "13:35", "San Francisco", "yes"]
    encoded_unknown = np.array(
        [
            label_encoders[i].transform([unknown_datapoint[i]])[0]
            for i in range(len(unknown_datapoint))
        ],
        dtype=float,
    ).reshape(1, -1)

    predicted_traffic = regressor.predict(encoded_unknown)[0]
    print(
        f"\nUnknown sample: {unknown_datapoint}\n"
        f"Predicted vehicles count: {predicted_traffic:.2f}"
    )

    # Compare with a known row to show realism.
    known_row_idx = 0
    print(
        f"\nKnown row example: {X_raw[known_row_idx].tolist()} "
        f"| actual={y[known_row_idx]:.0f}"
    )
    known_pred = regressor.predict(X_encoded[known_row_idx].reshape(1, -1))[0]
    print(f"Prediction for known row: {known_pred:.2f}")


if __name__ == "__main__":
    main()


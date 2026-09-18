# Copyright (c) 2026 Virtual Vehicle Research GmbH
# SPDX-License-Identifier: MIT
#
# Part of the Hybrid Model Generic Module developed within the SoliDAIR project.
# See LICENSE.md in the project root for the full license terms.

import io
import pickle
import time
import warnings
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
ANN_MODEL_NAME = "ANN"
ANN_INSTALL_GUIDANCE = "Install PyTorch separately in the environment where you run the app."


@dataclass
class RegressionTrainingResult:
    metrics_df: pd.DataFrame
    predictions_df: pd.DataFrame
    artifacts: Dict[str, bytes]
    details: Dict[str, object]


AVAILABLE_REGRESSION_MODELS = [
    "Random Forest",
    "Gradient Boosting Regressor",
    "MLP Regressor",
    ANN_MODEL_NAME,
]
THRESHOLD_BUCKETS = [
    "Below lower threshold",
    "Within thresholds",
    "Above upper threshold",
]
OUTSIDE_THRESHOLD_BUCKETS = {
    "Below lower threshold",
    "Above upper threshold",
}


def is_ann_available() -> bool:
    return find_spec("torch") is not None


def get_numeric_model_columns(df: pd.DataFrame) -> List[str]:
    numeric_columns = []
    for column in df.columns:
        if pd.to_numeric(df[column], errors="coerce").notna().any():
            numeric_columns.append(column)
    return numeric_columns


def prepare_regression_dataset(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, object]]:
    required_columns = feature_columns + [target_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required regression columns: {missing}")

    prepared = df[required_columns].apply(pd.to_numeric, errors="coerce")
    rows_before = len(prepared)
    prepared = prepared.dropna(subset=required_columns)
    rows_after = len(prepared)
    if rows_after < 10:
        raise ValueError(
            "At least 10 complete numeric rows are required for train/validation/test splitting."
        )

    X = prepared[feature_columns]
    y = prepared[target_column].astype(float)
    details = {
        "rows_before_dropna": rows_before,
        "rows_after_dropna": rows_after,
        "rows_dropped": rows_before - rows_after,
        "target_column": target_column,
        "feature_columns": feature_columns,
    }
    return X, y, details


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("Train/validation/test ratios must sum to a positive value.")

    train_ratio = train_ratio / ratio_sum
    val_ratio = val_ratio / ratio_sum
    test_ratio = test_ratio / ratio_sum
    holdout_ratio = val_ratio + test_ratio

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=holdout_ratio,
        random_state=random_state,
    )
    test_fraction_of_holdout = test_ratio / holdout_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=test_fraction_of_holdout,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def split_train_validation(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float,
    val_ratio: float,
    random_state: int,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_val_ratio_sum = train_ratio + val_ratio
    if train_val_ratio_sum <= 0:
        raise ValueError("Train and validation ratios must sum to a positive value.")

    validation_fraction = val_ratio / train_val_ratio_sum
    if lower is not None and upper is not None:
        X_train, X_val, y_train, y_val, _ = threshold_stratified_holdout_split(
            X,
            y,
            holdout_ratio=validation_fraction,
            random_state=random_state,
            lower=lower,
            upper=upper,
            singleton_outside_to_holdout=False,
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=validation_fraction,
            random_state=random_state,
        )
    return X_train, X_val, y_train, y_val


def threshold_bucket_labels(
    y: pd.Series,
    lower: float,
    upper: float,
) -> pd.Series:
    labels = pd.Series("Within thresholds", index=y.index, dtype="object")
    labels.loc[y < lower] = "Below lower threshold"
    labels.loc[y > upper] = "Above upper threshold"
    return labels


def threshold_bucket_summary(
    y: pd.Series,
    lower: Optional[float],
    upper: Optional[float],
    split_name: str,
) -> List[Dict[str, object]]:
    if lower is None or upper is None or len(y) == 0:
        return []

    labels = threshold_bucket_labels(y, lower, upper)
    total = len(labels)
    rows = []
    for bucket in THRESHOLD_BUCKETS:
        count = int((labels == bucket).sum())
        rows.append(
            {
                "split": split_name,
                "threshold_bucket": bucket,
                "count": count,
                "percentage": float(count / total * 100) if total else 0.0,
            }
        )
    outside_count = int(labels.isin(OUTSIDE_THRESHOLD_BUCKETS).sum())
    rows.append(
        {
            "split": split_name,
            "threshold_bucket": "Outside thresholds",
            "count": outside_count,
            "percentage": float(outside_count / total * 100) if total else 0.0,
        }
    )
    return rows


def threshold_stratified_holdout_split(
    X: pd.DataFrame,
    y: pd.Series,
    holdout_ratio: float,
    random_state: int,
    lower: float,
    upper: float,
    singleton_outside_to_holdout: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    if not 0 < holdout_ratio < 1:
        raise ValueError("Holdout ratio must be between 0 and 1.")

    labels = threshold_bucket_labels(
        y.reset_index(drop=True),
        lower,
        upper,
    )
    rng = np.random.default_rng(random_state)
    holdout_positions = []
    split_warnings = []

    for bucket in THRESHOLD_BUCKETS:
        bucket_positions = np.flatnonzero(labels.to_numpy() == bucket)
        bucket_count = len(bucket_positions)
        if bucket_count == 0:
            continue

        shuffled_positions = rng.permutation(bucket_positions)
        if bucket_count == 1:
            use_singleton_for_holdout = (
                singleton_outside_to_holdout and bucket in OUTSIDE_THRESHOLD_BUCKETS
            )
            holdout_count = 1 if use_singleton_for_holdout else 0
            if use_singleton_for_holdout:
                split_warnings.append(
                    f"Only one real-world row is {bucket.lower()}; it was reserved "
                    "for the test set so the model can be evaluated on that rare case."
                )
        else:
            holdout_count = int(round(bucket_count * holdout_ratio))
            holdout_count = max(1, holdout_count)
            holdout_count = min(holdout_count, bucket_count - 1)

        holdout_positions.extend(shuffled_positions[:holdout_count].tolist())

    if not holdout_positions:
        split_warnings.append(
            "Threshold-stratified splitting could not reserve a holdout row; "
            "falling back to a random split."
        )
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X,
            y,
            test_size=holdout_ratio,
            random_state=random_state,
        )
        return X_train, X_holdout, y_train, y_holdout, split_warnings

    train_mask = np.ones(len(X), dtype=bool)
    train_mask[holdout_positions] = False
    train_positions = np.flatnonzero(train_mask)
    if len(train_positions) == 0:
        split_warnings.append(
            "Threshold-stratified splitting would leave no training rows; "
            "falling back to a random split."
        )
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X,
            y,
            test_size=holdout_ratio,
            random_state=random_state,
        )
        return X_train, X_holdout, y_train, y_holdout, split_warnings

    return (
        X.iloc[train_positions],
        X.iloc[holdout_positions],
        y.iloc[train_positions],
        y.iloc[holdout_positions],
        split_warnings,
    )


def threshold_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    lower: Optional[float],
    upper: Optional[float],
) -> Dict[str, float]:
    if lower is None or upper is None:
        return {}

    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    true_outside = (y_true_array < lower) | (y_true_array > upper)
    pred_outside = (y_pred_array < lower) | (y_pred_array > upper)
    true_inside = ~true_outside
    pred_inside = ~pred_outside
    total = len(y_true_array)

    return {
        "true_positive_pct": float((true_outside & pred_outside).sum() / total * 100),
        "true_negative_pct": float((true_inside & pred_inside).sum() / total * 100),
        "false_positive_pct": float((true_inside & pred_outside).sum() / total * 100),
        "false_negative_pct": float((true_outside & pred_inside).sum() / total * 100),
    }


def regression_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    split: str,
    model_name: str,
    lower: Optional[float],
    upper: Optional[float],
) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics = {
        "model": model_name,
        "split": split,
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
    }
    metrics.update(threshold_metrics(y_true, y_pred, lower, upper))
    return metrics


def serialize_pickle_artifact(payload: object) -> bytes:
    return pickle.dumps(payload)


def make_safe_artifact_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def train_pytorch_ann(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: pd.Series,
    y_val: pd.Series,
    epochs: int,
    batch_size: int,
    patience: int,
    random_state: int,
) -> Tuple[object, List[Dict[str, float]], str]:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for the ANN model. "
            f"{ANN_INSTALL_GUIDANCE}"
        ) from exc

    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class RegressionModel(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 128):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 64),
                nn.SiLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            return self.model(x)

    model = RegressionModel(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train.to_numpy(dtype=float), dtype=torch.float32).view(-1, 1),
    )
    val_x = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_y = torch.tensor(y_val.to_numpy(dtype=float), dtype=torch.float32).view(-1, 1).to(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_state = None
    best_val_rmse = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * batch_x.size(0)

        train_rmse = float(np.sqrt(train_loss_sum / len(train_loader.dataset)))
        model.eval()
        with torch.no_grad():
            val_prediction = model(val_x)
            val_rmse = float(torch.sqrt(criterion(val_prediction, val_y)).cpu().item())

        history.append(
            {
                "epoch": epoch + 1,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
            }
        )

        if val_rmse < best_val_rmse - 1e-4:
            best_val_rmse = val_rmse
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, str(device)


def predict_pytorch_ann(model: object, X: np.ndarray) -> np.ndarray:
    import torch

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(X, dtype=torch.float32).to(device)
        return model(tensor).detach().cpu().numpy().reshape(-1)


def serialize_pytorch_artifact(payload: Dict[str, object]) -> bytes:
    import torch

    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def train_and_evaluate_regression_models(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    selected_model_names: Optional[List[str]] = None,
    production_df: Optional[pd.DataFrame] = None,
    simulation_df: Optional[pd.DataFrame] = None,
    thresholds: Optional[Dict[str, Dict[str, float]]] = None,
    training_dataset_label: str = "Hybrid",
    test_source_label: Optional[str] = None,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = RANDOM_STATE,
    random_forest_estimators: int = 100,
    mlp_max_iter: int = 1500,
    ann_epochs: int = 80,
    ann_batch_size: int = 64,
    ann_patience: int = 5,
) -> RegressionTrainingResult:
    selected_model_names = selected_model_names or [
        model_name
        for model_name in AVAILABLE_REGRESSION_MODELS
        if model_name != ANN_MODEL_NAME or is_ann_available()
    ]
    invalid_model_names = [
        model_name
        for model_name in selected_model_names
        if model_name not in AVAILABLE_REGRESSION_MODELS
    ]
    if invalid_model_names:
        raise ValueError(f"Unknown regression models selected: {', '.join(invalid_model_names)}")
    if not selected_model_names:
        raise ValueError("Select at least one regression model to train.")
    if ANN_MODEL_NAME in selected_model_names and not is_ann_available():
        raise ImportError(
            "ANN requires PyTorch, which is not part of the default dependencies. "
            f"{ANN_INSTALL_GUIDANCE}"
        )

    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("Train/validation/test ratios must sum to a positive value.")
    effective_train_ratio = train_ratio / ratio_sum
    effective_val_ratio = val_ratio / ratio_sum
    effective_test_ratio = test_ratio / ratio_sum

    threshold = thresholds.get(target_column) if thresholds else None
    lower = float(threshold["lower"]) if threshold else None
    upper = float(threshold["upper"]) if threshold else None
    target_suffix = make_safe_artifact_name(target_column)
    dataset_suffix = make_safe_artifact_name(training_dataset_label.lower().replace(" ", "_"))
    split_warnings: List[str] = []
    split_strategy = "random"

    uses_production_only_test = production_df is not None and simulation_df is not None
    if uses_production_only_test:
        X_production, y_production, production_details = prepare_regression_dataset(
            production_df,
            target_column,
            feature_columns,
        )
        if simulation_df.empty:
            X_simulation = pd.DataFrame(columns=X_production.columns)
            y_simulation = pd.Series(dtype=float)
            simulation_details = {
                "rows_before_dropna": 0,
                "rows_after_dropna": 0,
                "rows_dropped": 0,
            }
        else:
            X_simulation, y_simulation, simulation_details = prepare_regression_dataset(
                simulation_df,
                target_column,
                feature_columns,
            )
        if lower is not None and upper is not None:
            (
                X_production_train_val,
                X_test,
                y_production_train_val,
                y_test,
                test_split_warnings,
            ) = threshold_stratified_holdout_split(
                X_production,
                y_production,
                holdout_ratio=effective_test_ratio,
                random_state=random_state,
                lower=lower,
                upper=upper,
                singleton_outside_to_holdout=True,
            )
            split_strategy = "threshold_stratified_production_test"
            split_warnings.extend(test_split_warnings)
        else:
            X_production_train_val, X_test, y_production_train_val, y_test = train_test_split(
                X_production,
                y_production,
                test_size=effective_test_ratio,
                random_state=random_state,
            )
            split_strategy = "random_production_test"

        if X_simulation.empty:
            X_train_val = X_production_train_val.reset_index(drop=True)
            y_train_val = y_production_train_val.reset_index(drop=True)
        else:
            X_train_val = pd.concat(
                [X_production_train_val, X_simulation],
                axis=0,
                ignore_index=True,
            )
            y_train_val = pd.concat(
                [y_production_train_val, y_simulation],
                axis=0,
                ignore_index=True,
            )
        X_train, X_val, y_train, y_val = split_train_validation(
            X_train_val,
            y_train_val,
            effective_train_ratio,
            effective_val_ratio,
            random_state,
            lower=lower,
            upper=upper,
        )
        preparation_details = {
            "rows_before_dropna": int(
                production_details["rows_before_dropna"] + simulation_details["rows_before_dropna"]
            ),
            "rows_after_dropna": int(
                production_details["rows_after_dropna"] + simulation_details["rows_after_dropna"]
            ),
            "rows_dropped": int(
                production_details["rows_dropped"] + simulation_details["rows_dropped"]
            ),
            "target_column": target_column,
            "feature_columns": feature_columns,
            "test_source": "production_only",
            "production_rows_after_dropna": int(production_details["rows_after_dropna"]),
            "simulation_rows_after_dropna": int(simulation_details["rows_after_dropna"]),
            "production_rows_reserved_for_test": int(X_test.shape[0]),
            "production_rows_available_for_train_validation": int(X_production_train_val.shape[0]),
            "simulation_rows_available_for_train_validation": int(X_simulation.shape[0]),
        }
    else:
        single_dataset_source = (
            "production" if test_source_label in {"Production", "Real-world"} else "hybrid"
        )
        X, y, preparation_details = prepare_regression_dataset(
            df,
            target_column,
            feature_columns,
        )
        if lower is not None and upper is not None:
            X_train_val, X_test, y_train_val, y_test, test_split_warnings = (
                threshold_stratified_holdout_split(
                    X,
                    y,
                    holdout_ratio=effective_test_ratio,
                    random_state=random_state,
                    lower=lower,
                    upper=upper,
                    singleton_outside_to_holdout=True,
                )
            )
            X_train, X_val, y_train, y_val = split_train_validation(
                X_train_val,
                y_train_val,
                effective_train_ratio,
                effective_val_ratio,
                random_state,
                lower=lower,
                upper=upper,
            )
            split_strategy = f"threshold_stratified_{single_dataset_source}_split"
            split_warnings.extend(test_split_warnings)
        else:
            X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
                X,
                y,
                train_ratio,
                val_ratio,
                test_ratio,
                random_state,
            )
            split_strategy = f"random_{single_dataset_source}_split"
        preparation_details["test_source"] = (
            f"{single_dataset_source}_threshold_stratified_split"
            if lower is not None and upper is not None
            else f"{single_dataset_source}_random_split"
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model_specs = [
        (
            "Random Forest",
            RandomForestRegressor(
                n_estimators=random_forest_estimators,
                random_state=random_state,
                n_jobs=-1,
            ),
            X_train,
            X_val,
            X_test,
            False,
        ),
        (
            "Gradient Boosting Regressor",
            GradientBoostingRegressor(random_state=random_state),
            X_train,
            X_val,
            X_test,
            False,
        ),
        (
            "MLP Regressor",
            MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                learning_rate="adaptive",
                random_state=random_state,
                max_iter=mlp_max_iter,
                early_stopping=True,
                validation_fraction=0.1,
            ),
            X_train_scaled,
            X_val_scaled,
            X_test_scaled,
            True,
        ),
    ]
    model_specs = [
        model_spec for model_spec in model_specs if model_spec[0] in selected_model_names
    ]

    metrics = []
    artifacts: Dict[str, bytes] = {}
    prediction_prefix = f"{training_dataset_label} - "
    resolved_test_source_label = test_source_label or (
        "Real-world" if uses_production_only_test else "Hybrid"
    )
    prediction_rows = pd.DataFrame(
        {
            "original_index": y_test.index.astype(str),
            "source": resolved_test_source_label,
            "y_true": y_test.to_numpy(dtype=float),
        }
    )
    if lower is not None and upper is not None:
        prediction_rows["threshold_bucket"] = threshold_bucket_labels(
            y_test,
            lower,
            upper,
        ).to_numpy()

    threshold_split_summary = (
        threshold_bucket_summary(y_train, lower, upper, "train")
        + threshold_bucket_summary(y_val, lower, upper, "validation")
        + threshold_bucket_summary(y_test, lower, upper, "test")
    )
    training_details: Dict[str, object] = {
        **preparation_details,
        "train_shape": list(X_train.shape),
        "validation_shape": list(X_val.shape),
        "test_shape": list(X_test.shape),
        "train_ratio": effective_train_ratio,
        "validation_ratio": effective_val_ratio,
        "test_ratio": effective_test_ratio,
        "random_state": random_state,
        "thresholds": threshold,
        "selected_model_names": selected_model_names,
        "training_dataset": training_dataset_label,
        "test_source_label": resolved_test_source_label,
        "split_strategy": split_strategy,
        "split_warnings": split_warnings,
        "model_training_warnings": [],
        "threshold_split_summary": threshold_split_summary,
    }

    for model_name, model, train_x, val_x, test_x, uses_scaler in model_specs:
        start = time.time()
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(train_x, y_train)
        elapsed = time.time() - start
        for caught_warning in caught_warnings:
            if issubclass(caught_warning.category, ConvergenceWarning):
                training_details["model_training_warnings"].append(
                    {
                        "training_dataset": training_dataset_label,
                        "model": model_name,
                        "warning": str(caught_warning.message),
                    }
                )
        for split_name, split_x, split_y in [
            ("train", train_x, y_train),
            ("validation", val_x, y_val),
            ("test", test_x, y_test),
        ]:
            y_pred = model.predict(split_x)
            record = regression_metrics(split_y, y_pred, split_name, model_name, lower, upper)
            record["training_dataset"] = training_dataset_label
            record["training_seconds"] = elapsed if split_name == "train" else np.nan
            metrics.append(record)

        test_prediction = model.predict(test_x)
        prediction_rows[f"{prediction_prefix}{model_name} prediction"] = test_prediction
        artifact_payload = {
            "model": model,
            "scaler": scaler if uses_scaler else None,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "uses_scaler": uses_scaler,
            "training_details": training_details,
        }
        model_suffix = make_safe_artifact_name(model_name.lower().replace(" ", "_"))
        artifact_filename = f"{dataset_suffix}_{model_suffix}_{target_suffix}.pkl"
        artifacts[artifact_filename] = serialize_pickle_artifact(artifact_payload)

    uses_scaled_features = any(model_name == "MLP Regressor" for model_name in selected_model_names)
    if ANN_MODEL_NAME in selected_model_names:
        ann_start = time.time()
        ann_model, ann_history, ann_device = train_pytorch_ann(
            X_train_scaled,
            X_val_scaled,
            y_train,
            y_val,
            epochs=ann_epochs,
            batch_size=ann_batch_size,
            patience=ann_patience,
            random_state=random_state,
        )
        ann_elapsed = time.time() - ann_start
        for split_name, split_x, split_y in [
            ("train", X_train_scaled, y_train),
            ("validation", X_val_scaled, y_val),
            ("test", X_test_scaled, y_test),
        ]:
            y_pred = predict_pytorch_ann(ann_model, split_x)
            record = regression_metrics(split_y, y_pred, split_name, "ANN", lower, upper)
            record["training_dataset"] = training_dataset_label
            record["training_seconds"] = ann_elapsed if split_name == "train" else np.nan
            metrics.append(record)

        ann_test_prediction = predict_pytorch_ann(ann_model, X_test_scaled)
        prediction_rows[f"{prediction_prefix}ANN prediction"] = ann_test_prediction
        artifacts[f"{dataset_suffix}_ann_{target_suffix}.pt"] = serialize_pytorch_artifact(
            {
                "state_dict": {
                    key: value.detach().cpu() for key, value in ann_model.state_dict().items()
                },
                "scaler": scaler,
                "input_dim": X_train.shape[1],
                "target_column": target_column,
                "feature_columns": feature_columns,
                "history": ann_history,
                "device_used": ann_device,
                "training_details": training_details,
            }
        )
        training_details["ann_history"] = ann_history
        training_details["ann_device"] = ann_device
        uses_scaled_features = True

    if uses_scaled_features:
        artifacts[f"{dataset_suffix}_standard_scaler_{target_suffix}.pkl"] = (
            serialize_pickle_artifact(
                {
                    "scaler": scaler,
                    "target_column": target_column,
                    "feature_columns": feature_columns,
                }
            )
        )

    metrics_df = pd.DataFrame(metrics)
    return RegressionTrainingResult(
        metrics_df=metrics_df,
        predictions_df=prediction_rows,
        artifacts=artifacts,
        details=training_details,
    )

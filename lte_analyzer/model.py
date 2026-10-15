"""Predict downlink throughput from RSRP, RSRQ, and SINR.

The model is ordinary linear regression. The baseline predicts the training-set
average on every test row. A time holdout — the last 25% of each route — is
used instead of a random split, because neighboring drive-test samples leak
into each other when you shuffle them.

The fit is a description of this log. It is not a link budget and not a
capacity model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

BASE_FEATURES = ("rsrp", "rsrq", "sinr")
FEATURE_UNITS = {"rsrp": "dBm", "rsrq": "dB", "sinr": "dB", "bw_mhz": "MHz"}
FEATURE_LABELS = {"rsrp": "RSRP", "rsrq": "RSRQ", "sinr": "SINR", "bw_mhz": "Bandwidth"}


def feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = list(BASE_FEATURES)
    if "bw_mhz" in frame.columns and frame["bw_mhz"].notna().sum() >= 30 and frame["bw_mhz"].nunique(dropna=True) > 1:
        columns.append("bw_mhz")
    return columns


@dataclass
class ModelResult:
    ok: bool
    reason: str
    n_train: int = 0
    n_test: int = 0
    routes_train_only: list[str] = field(default_factory=list)
    baseline_mae: float | None = None
    baseline_rmse: float | None = None
    baseline_r2: float | None = None
    model_mae: float | None = None
    model_rmse: float | None = None
    model_r2: float | None = None
    train_r2: float | None = None
    intercept: float | None = None
    coefficients: list[dict[str, float | str]] = field(default_factory=list)
    y_test: list[float] = field(default_factory=list)
    pred_test: list[float] = field(default_factory=list)
    note: str = ""
    split_note: str = ""


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(r2_score(y_true, y_pred))
    return mae, rmse, r2


def _holdout(frame: pd.DataFrame, frac: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    train_only: list[str] = []
    for route, group in frame.groupby("route", sort=False):
        ordered = group.sort_values("timestamp")
        n = len(ordered)
        k = max(1, int(round(n * frac)))
        if n - k < 15 or k < 8:
            train_parts.append(ordered)
            train_only.append(str(route))
            continue
        train_parts.append(ordered.iloc[:-k])
        test_parts.append(ordered.iloc[-k:])
    train = pd.concat(train_parts, ignore_index=True) if train_parts else frame.iloc[0:0]
    test = pd.concat(test_parts, ignore_index=True) if test_parts else frame.iloc[0:0]
    return train, test, train_only


def fit_throughput(frame: pd.DataFrame) -> ModelResult:
    features = feature_columns(frame)
    needed = frame.dropna(subset=features + ["dl_mbps", "timestamp", "route"]).copy()
    if len(needed) < 40:
        return ModelResult(
            ok=False,
            reason=(
                f"Only {len(needed)} complete samples. The linear model needs a longer drive "
                "(about 40 or more rows with RSRP, RSRQ, SINR, and downlink)."
            ),
        )

    train, test, train_only = _holdout(needed)
    if len(test) < 15 or len(train) < 20:
        return ModelResult(
            ok=False,
            reason=(
                "The time holdout did not leave enough rows in both sides. "
                "Add another route, or a longer file, and fit again."
            ),
            n_train=int(len(train)),
            n_test=int(len(test)),
            routes_train_only=train_only,
        )

    x_train = train.loc[:, features].to_numpy(dtype=float)
    y_train = train["dl_mbps"].to_numpy(dtype=float)
    x_test = test.loc[:, features].to_numpy(dtype=float)
    y_test = test["dl_mbps"].to_numpy(dtype=float)

    baseline_pred = np.full_like(y_test, fill_value=float(y_train.mean()))
    base_mae, base_rmse, base_r2 = _metrics(y_test, baseline_pred)

    reg = LinearRegression()
    reg.fit(x_train, y_train)
    pred = reg.predict(x_test)
    model_mae, model_rmse, model_r2 = _metrics(y_test, pred)
    train_r2 = float(r2_score(y_train, reg.predict(x_train)))

    stds = x_train.std(axis=0)
    coefficients: list[dict[str, float | str]] = []
    for name, coef, std in zip(features, reg.coef_, stds, strict=True):
        coefficients.append(
            {
                "feature": FEATURE_LABELS[name],
                "column": name,
                "unit": FEATURE_UNITS[name],
                "coef": float(coef),
                "std": float(std),
                "per_std": float(coef * std),
            }
        )

    note = _note(
        base_mae=base_mae,
        model_mae=model_mae,
        model_r2=model_r2,
        train_r2=train_r2,
        coefficients=coefficients,
        sinr_rsrq_corr=_pair_corr(train, "sinr", "rsrq"),
    )
    split_note = (
        "Each route is sorted by time. The last 25% of its samples are the test set, "
        "and the earlier samples are the training set. A random split would look better "
        "and would be less honest, because a point and its neighbor a few seconds later "
        "are almost the same measurement."
    )
    if train_only:
        split_note += " These routes were too short to hold out and stayed in training only: " + ", ".join(train_only) + "."

    return ModelResult(
        ok=True,
        reason="",
        n_train=int(len(train)),
        n_test=int(len(test)),
        routes_train_only=train_only,
        baseline_mae=base_mae,
        baseline_rmse=base_rmse,
        baseline_r2=base_r2,
        model_mae=model_mae,
        model_rmse=model_rmse,
        model_r2=model_r2,
        train_r2=train_r2,
        intercept=float(reg.intercept_),
        coefficients=coefficients,
        y_test=[float(v) for v in y_test],
        pred_test=[float(v) for v in pred],
        note=note,
        split_note=split_note,
    )


def _pair_corr(frame: pd.DataFrame, left: str, right: str) -> float | None:
    pair = frame[[left, right]].dropna()
    if len(pair) < 20 or pair[left].nunique() < 2:
        return None
    value = float(pair[left].corr(pair[right]))
    return None if np.isnan(value) else value


def _note(
    base_mae: float,
    model_mae: float,
    model_r2: float,
    train_r2: float,
    coefficients: list[dict[str, float | str]],
    sinr_rsrq_corr: float | None,
) -> str:
    parts: list[str] = []
    if model_mae < base_mae * 0.97:
        lift = base_mae - model_mae
        r2_sentence = f"Test R² is {model_r2:.2f}. Training R² is {train_r2:.2f}."
        if model_r2 < 0:
            r2_sentence = (
                f"Test R² is {model_r2:.2f}, below zero, so the line still misses how spread out "
                f"the held-out speeds are. Training R² is {train_r2:.2f}. "
                "Judge this fit by the MAE: typical error in Mbps, not by a flattering R²."
            )
        parts.append(
            f"On the held-out minutes, linear regression cut the typical error from "
            f"{base_mae:.1f} Mbps to {model_mae:.1f} Mbps versus guessing the training-set average "
            f"(about {lift:.1f} Mbps closer). {r2_sentence}"
        )
    elif model_mae > base_mae * 1.02:
        parts.append(
            f"On the held-out minutes, the linear model was worse than guessing the average "
            f"({model_mae:.1f} Mbps of typical error versus {base_mae:.1f}). "
            "The radio KPIs in this split do not carry a stable linear story about speed."
        )
    else:
        parts.append(
            f"On the held-out minutes, the linear model and the average-throughput guess are close "
            f"({model_mae:.1f} versus {base_mae:.1f} Mbps of typical error). "
            "There is not a clear linear win here."
        )

    ranked = sorted(coefficients, key=lambda item: abs(float(item["per_std"])), reverse=True)
    lead = ranked[0]
    parts.append(
        f"The largest typical swing is {lead['feature']}: one standard deviation "
        f"({float(lead['std']):.1f} {lead['unit']}) lines up with about {float(lead['per_std']):.1f} Mbps "
        "in this fit, holding the other inputs fixed on paper. "
        "That is an association inside the equation, not a dial you can turn in the field."
    )
    rsrp_item = next(item for item in coefficients if item["column"] == "rsrp")
    sinr_item = next(item for item in coefficients if item["column"] == "sinr")
    if float(rsrp_item["coef"]) <= 0 and abs(float(sinr_item["per_std"])) > abs(float(rsrp_item["per_std"])):
        parts.append(
            "RSRP's own coefficient is small or negative once SINR is in the equation. "
            "The two move together, so the fit hands the speed story to SINR. "
            "That is not evidence that a louder cell is slower."
        )
    if any(item["column"] == "bw_mhz" for item in coefficients):
        parts.append(
            "Channel bandwidth is in the fit because this log recorded it. "
            "A 40 MHz 5G sample and a 10 MHz LTE sample are not the same pipe, even at the same SINR. "
            "A file without a bandwidth column is fit on RSRP, RSRQ, and SINR only."
        )
    if sinr_rsrq_corr is not None and abs(sinr_rsrq_corr) >= 0.75:
        parts.append(
            f"SINR and RSRQ move together on the training rows (r = {sinr_rsrq_corr:.2f}), "
            "so their separate coefficients are shaky. Trust the joint point — link quality tracks speed — "
            "more than the exact Mbps-per-dB split between them."
        )
    parts.append(
        "Do not use this score for planning. The test minutes are later on the same drive, in the same city, "
        "with the same phone behavior baked into the log. A new route, a new band, or a loaded cell can miss this line by a lot. "
        "The model also ignores bandwidth, MIMO layers, and whether anyone was actually sending traffic."
    )
    return " ".join(parts)

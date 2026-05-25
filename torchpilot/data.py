"""Raw data loading and per-spec preprocessing.

Data flow:
    load_raw(csv)          -> RawData (integer-encoded cats, raw numerics, splits)
    Preprocessor(spec).fit(raw) -> ready to transform numerics + targets for training
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from torch.utils.data import DataLoader, TensorDataset

from torchpilot.schemas import ArchSpec


Task = Literal["classification", "regression"]


@dataclass
class RawData:
    """Data after splitting + categorical integer-encoding, but before any scaling."""
    X_num_train: np.ndarray
    X_num_val: np.ndarray
    X_cat_train: np.ndarray  # int64; shape (N, n_cat) — 0 if no categorical cols
    X_cat_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray

    numeric_cols: list[str]
    categorical_cols: list[str]
    cat_cardinalities: list[int]
    cat_value_to_idx: list[dict[str, int]]  # per-categorical-col mapping for predict()
    label_encoder: LabelEncoder | None      # classification target encoder
    task: Task
    n_outputs: int
    target_name: str

    @property
    def n_numeric(self) -> int:
        return self.X_num_train.shape[1]

    @property
    def n_categorical(self) -> int:
        return len(self.categorical_cols)


def load_raw(
    csv_path: str,
    target: str,
    task: Task,
    val_size: float = 0.2,
    random_state: int = 42,
) -> RawData:
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not in CSV (columns: {list(df.columns)}).")

    df = df.dropna(subset=[target]).reset_index(drop=True)
    y_raw = df[target]
    X_df = df.drop(columns=[target])

    numeric_cols = X_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X_df.columns if c not in numeric_cols]

    # Median impute numerics, "__missing__" token for categoricals.
    for c in numeric_cols:
        X_df[c] = X_df[c].fillna(X_df[c].median())
    for c in categorical_cols:
        X_df[c] = X_df[c].fillna("__missing__").astype(str)

    # Integer-encode categoricals with an explicit OOV slot at index 0.
    cat_value_to_idx: list[dict[str, int]] = []
    cat_cardinalities: list[int] = []
    X_cat_full = np.zeros((len(X_df), len(categorical_cols)), dtype=np.int64)
    for j, c in enumerate(categorical_cols):
        uniques = X_df[c].unique().tolist()
        mapping = {"__oov__": 0}
        for v in uniques:
            if v not in mapping:
                mapping[v] = len(mapping)
        cat_value_to_idx.append(mapping)
        cat_cardinalities.append(len(mapping))
        X_cat_full[:, j] = X_df[c].map(mapping).fillna(0).astype(np.int64).to_numpy()

    X_num_full = (
        X_df[numeric_cols].to_numpy(dtype=np.float32)
        if numeric_cols
        else np.zeros((len(X_df), 0), dtype=np.float32)
    )

    label_encoder: LabelEncoder | None = None
    if task == "classification":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw).astype(np.int64)
        n_outputs = len(label_encoder.classes_)
        stratify = y
    else:
        y = y_raw.to_numpy(dtype=np.float32)
        n_outputs = 1
        stratify = None

    indices = np.arange(len(X_df))
    tr_idx, va_idx = train_test_split(
        indices, test_size=val_size, random_state=random_state, stratify=stratify
    )

    return RawData(
        X_num_train=X_num_full[tr_idx],
        X_num_val=X_num_full[va_idx],
        X_cat_train=X_cat_full[tr_idx],
        X_cat_val=X_cat_full[va_idx],
        y_train=y[tr_idx],
        y_val=y[va_idx],
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        cat_cardinalities=cat_cardinalities,
        cat_value_to_idx=cat_value_to_idx,
        label_encoder=label_encoder,
        task=task,
        n_outputs=n_outputs,
        target_name=target,
    )


def profile(raw: RawData, csv_path: str) -> dict:
    """Dataset profile for the LLM. Computed off the raw CSV for richer summary stats."""
    df = pd.read_csv(csv_path)
    feature_summary: dict[str, dict] = {}
    for c in raw.numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        feature_summary[c] = {
            "dtype": "numeric",
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "missing_pct": float(df[c].isna().mean()),
        }
    for c in raw.categorical_cols:
        s = df[c].dropna().astype(str)
        feature_summary[c] = {
            "dtype": "categorical",
            "cardinality": int(s.nunique()),
            "top_values": s.value_counts().head(5).to_dict(),
            "missing_pct": float(df[c].isna().mean()),
        }

    target_info: dict = {"name": raw.target_name}
    if raw.task == "classification":
        counts = df[raw.target_name].value_counts(normalize=True).round(3).to_dict()
        target_info["n_classes"] = int(raw.n_outputs)
        target_info["class_balance"] = {str(k): float(v) for k, v in counts.items()}
    else:
        s = pd.to_numeric(df[raw.target_name], errors="coerce").dropna()
        target_info["min"] = float(s.min())
        target_info["max"] = float(s.max())
        target_info["mean"] = float(s.mean())
        target_info["std"] = float(s.std())

    return {
        "task": raw.task,
        "n_rows": int(len(df)),
        "n_numeric_columns": len(raw.numeric_cols),
        "n_categorical_columns": len(raw.categorical_cols),
        "categorical_cardinalities": dict(zip(raw.categorical_cols, raw.cat_cardinalities)),
        "target": target_info,
        "features": feature_summary,
    }


# ---------- per-spec preprocessing ----------

_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


@dataclass
class Preprocessor:
    spec: ArchSpec
    scaler: object | None = None
    target_mean: float = 0.0
    target_std: float = 1.0

    def fit(self, raw: RawData) -> "Preprocessor":
        if raw.n_numeric > 0 and self.spec.numeric_scaler != "none":
            self.scaler = _SCALERS[self.spec.numeric_scaler]()
            self.scaler.fit(raw.X_num_train)
        if raw.task == "regression" and self.spec.target_standardize:
            self.target_mean = float(raw.y_train.mean())
            self.target_std = float(raw.y_train.std() or 1.0)
        return self

    def _transform_numeric(self, X_num: np.ndarray) -> np.ndarray:
        if X_num.shape[1] == 0:
            return X_num
        if self.scaler is not None:
            return self.scaler.transform(X_num).astype(np.float32)
        return X_num.astype(np.float32)

    def transform_split(self, raw: RawData, split: Literal["train", "val"]) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        X_num = raw.X_num_train if split == "train" else raw.X_num_val
        X_cat = raw.X_cat_train if split == "train" else raw.X_cat_val
        y = raw.y_train if split == "train" else raw.y_val
        X_num_t = torch.from_numpy(self._transform_numeric(X_num)).float()
        X_cat_t = torch.from_numpy(X_cat).long()
        if raw.task == "classification":
            y_t = torch.from_numpy(y).long()
        else:
            y_arr = (y - self.target_mean) / self.target_std if self.spec.target_standardize else y
            y_t = torch.from_numpy(y_arr.astype(np.float32))
        return X_num_t, X_cat_t, y_t

    def transform_new(self, X_num_raw: np.ndarray, X_cat_raw: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self._transform_numeric(X_num_raw)).float(),
            torch.from_numpy(X_cat_raw).long(),
        )

    def inverse_target(self, y: np.ndarray) -> np.ndarray:
        if not self.spec.target_standardize:
            return y
        return y * self.target_std + self.target_mean


def loaders_from_preprocessor(
    raw: RawData, prep: Preprocessor, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    X_num_tr, X_cat_tr, y_tr = prep.transform_split(raw, "train")
    X_num_va, X_cat_va, y_va = prep.transform_split(raw, "val")
    train_ds = TensorDataset(X_num_tr, X_cat_tr, y_tr)
    val_ds = TensorDataset(X_num_va, X_cat_va, y_va)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )

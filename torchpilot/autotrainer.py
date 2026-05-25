from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch

from torchpilot.data import Preprocessor, RawData, load_raw, profile
from torchpilot.llm import LLMArchitect
from torchpilot.metrics import primary_better, primary_metric_name
from torchpilot.model import TabularNet
from torchpilot.report import write_report
from torchpilot.schemas import ArchSpec, TrialResult
from torchpilot.trainer import train


Task = Literal["classification", "regression"]


CHECKPOINT_VERSION = 1


@dataclass
class FitReport:
    best_spec: ArchSpec
    best_result: TrialResult
    history: list[TrialResult]
    report_path: Path | None = None
    model_path: Path | None = None


class TorchPilot:
    """Iterative LLM-driven AutoML for tabular PyTorch models.

    Example:
        auto = TorchPilot("data.csv", target="species", task="classification")
        report = auto.fit(n_rounds=5, report_path="run.pdf", save_path="model.pt")
        preds = auto.predict(new_df)

        # Later, in a different process:
        auto = TorchPilot.load("model.pt")
        preds = auto.predict(new_df)
    """

    def __init__(
        self,
        csv_path: str,
        target: str,
        task: Task,
        llm_model: str = "gpt-oss:120b-cloud",
        api_key: str | None = None,
        val_size: float = 0.2,
        random_state: int = 42,
        device: str | None = None,
    ):
        self.csv_path = csv_path
        self.target = target
        self.task = task
        self.llm_model = llm_model
        self.device = device
        self.raw: RawData = load_raw(
            csv_path, target=target, task=task, val_size=val_size, random_state=random_state
        )
        self.profile: dict = profile(self.raw, csv_path)
        self.architect: LLMArchitect | None = LLMArchitect(model=llm_model, api_key=api_key)
        self.history: list[TrialResult] = []
        self.best_model: TabularNet | None = None
        self.best_preprocessor: Preprocessor | None = None
        self.best_spec: ArchSpec | None = None
        self.best_result: TrialResult | None = None

    def fit(
        self,
        n_rounds: int = 5,
        patience: int = 10,
        verbose: bool = True,
        report_path: str | Path | None = None,
        save_path: str | Path | None = None,
    ) -> FitReport:
        if self.architect is None:
            raise RuntimeError(
                "This instance was loaded from a checkpoint; fit() requires the LLM architect. "
                "Construct a fresh TorchPilot(csv_path, target, task, ...) to retrain."
            )
        primary = primary_metric_name(self.task)
        for r in range(1, n_rounds + 1):
            if verbose:
                print(f"\n=== Round {r}/{n_rounds} — asking LLM for a configuration ===")
            spec = self.architect.propose(self.profile, self.history)
            if verbose:
                self._print_spec(spec)

            model, prep, outcome = train(
                spec, self.raw, patience=patience, device=self.device, verbose=verbose
            )
            result = TrialResult(
                round=r,
                spec=spec,
                train_loss=outcome.train_loss_history[-1] if outcome.train_loss_history else float("nan"),
                val_loss=outcome.best_val_loss,
                val_metric=outcome.primary_val_metric,
                metric_name=outcome.metric_name,
                epochs_run=outcome.epochs_run,
                final_metrics=outcome.final_metrics,
                train_loss_history=outcome.train_loss_history,
                val_loss_history=outcome.val_loss_history,
                val_metric_history=outcome.val_metric_history,
                trajectory=outcome.trajectory,
            )
            self.history.append(result)
            if verbose:
                metric_strs = " ".join(f"{k}={v:.4f}" for k, v in outcome.final_metrics.items())
                print(
                    f"Round {r} → val_loss={result.val_loss:.4f} | {metric_strs} "
                    f"| trajectory: {outcome.trajectory} ({outcome.epochs_run} epochs)"
                )

            if (
                self.best_result is None
                or primary_better(self.task, result.val_metric, self.best_result.val_metric)
            ):
                self.best_model = model
                self.best_preprocessor = prep
                self.best_spec = spec
                self.best_result = result
                if verbose:
                    print(f"  ↑ new best (val {primary}={result.val_metric:.4f})")

        assert self.best_result is not None and self.best_spec is not None

        out_path: Path | None = None
        if report_path is not None:
            out_path = write_report(
                output_path=report_path,
                profile=self.profile,
                history=self.history,
                best_spec=self.best_spec,
                best_result=self.best_result,
                csv_path=self.csv_path,
                llm_model=self.llm_model,
            )
            if verbose:
                print(f"\nPDF report written to: {out_path}")

        model_out_path: Path | None = None
        if save_path is not None:
            model_out_path = self.save(save_path)
            if verbose:
                print(f"Best model saved to: {model_out_path}")

        return FitReport(
            best_spec=self.best_spec,
            best_result=self.best_result,
            history=list(self.history),
            report_path=out_path,
            model_path=model_out_path,
        )

    # ---------- persistence ----------

    def save(self, path: str | Path) -> Path:
        """Serialize the best model + preprocessor + metadata to a single .pt file."""
        if self.best_model is None or self.best_spec is None or self.best_preprocessor is None:
            raise RuntimeError("Call fit() before save() — there is no best model yet.")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "version": CHECKPOINT_VERSION,
            "task": self.task,
            "target": self.target,
            "csv_path": str(self.csv_path),
            "llm_model": self.llm_model,
            "best_spec": self.best_spec.model_dump(),
            "model_state_dict": {
                k: v.detach().cpu() for k, v in self.best_model.state_dict().items()
            },
            "preprocessor": self.best_preprocessor,
            "raw_meta": {
                "numeric_cols": self.raw.numeric_cols,
                "categorical_cols": self.raw.categorical_cols,
                "cat_cardinalities": self.raw.cat_cardinalities,
                "cat_value_to_idx": self.raw.cat_value_to_idx,
                "n_outputs": self.raw.n_outputs,
                "label_encoder": self.raw.label_encoder,
            },
            "profile": self.profile,
            "best_metrics": (self.best_result.final_metrics if self.best_result else {}),
        }
        torch.save(checkpoint, p)
        return p

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "TorchPilot":
        """Reconstruct a predict-ready TorchPilot from a checkpoint produced by save()."""
        ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
        if ckpt.get("version") != CHECKPOINT_VERSION:
            raise RuntimeError(
                f"Checkpoint version {ckpt.get('version')} != expected {CHECKPOINT_VERSION}."
            )

        meta = ckpt["raw_meta"]
        n_num = len(meta["numeric_cols"])
        n_cat = len(meta["categorical_cols"])

        inst = cls.__new__(cls)
        inst.csv_path = ckpt.get("csv_path", "<loaded>")
        inst.target = ckpt["target"]
        inst.task = ckpt["task"]
        inst.llm_model = ckpt.get("llm_model", "")
        inst.device = device
        inst.history = []
        inst.profile = ckpt.get("profile", {})
        inst.architect = None  # fit() blocked from a loaded instance

        # Synthetic empty splits — predict() only reads the metadata fields.
        inst.raw = RawData(
            X_num_train=np.zeros((0, n_num), dtype=np.float32),
            X_num_val=np.zeros((0, n_num), dtype=np.float32),
            X_cat_train=np.zeros((0, n_cat), dtype=np.int64),
            X_cat_val=np.zeros((0, n_cat), dtype=np.int64),
            y_train=np.zeros(0, dtype=np.float32),
            y_val=np.zeros(0, dtype=np.float32),
            numeric_cols=meta["numeric_cols"],
            categorical_cols=meta["categorical_cols"],
            cat_cardinalities=meta["cat_cardinalities"],
            cat_value_to_idx=meta["cat_value_to_idx"],
            label_encoder=meta["label_encoder"],
            task=ckpt["task"],
            n_outputs=meta["n_outputs"],
            target_name=ckpt["target"],
        )

        inst.best_spec = ArchSpec(**ckpt["best_spec"])
        inst.best_preprocessor = ckpt["preprocessor"]
        inst.best_result = None  # not preserved; metrics live in ckpt["best_metrics"] if needed

        dev = torch.device(device or "cpu")
        inst.best_model = TabularNet(
            inst.best_spec,
            inst.raw.n_numeric,
            inst.raw.cat_cardinalities,
            inst.raw.n_outputs,
        )
        inst.best_model.load_state_dict(ckpt["model_state_dict"])
        inst.best_model.to(dev).eval()
        return inst

    def write_report(self, output_path: str | Path) -> Path:
        if self.best_result is None or self.best_spec is None:
            raise RuntimeError("Call fit() before write_report().")
        return write_report(
            output_path=output_path,
            profile=self.profile,
            history=self.history,
            best_spec=self.best_spec,
            best_result=self.best_result,
            csv_path=self.csv_path,
            llm_model=self.llm_model,
        )

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.best_model is None or self.best_preprocessor is None:
            raise RuntimeError("Call fit() (or load()) before predict().")
        x_num_raw, x_cat_raw = self._encode_new(df)
        x_num_t, x_cat_t = self.best_preprocessor.transform_new(x_num_raw, x_cat_raw)
        dev = next(self.best_model.parameters()).device
        self.best_model.eval()
        with torch.no_grad():
            out = self.best_model(x_num_t.to(dev), x_cat_t.to(dev))
            if self.task == "classification":
                idx = out.argmax(dim=1).cpu().numpy()
                if self.raw.label_encoder is not None:
                    return self.raw.label_encoder.inverse_transform(idx)
                return idx
            preds = out.squeeze(-1).cpu().numpy()
            return self.best_preprocessor.inverse_target(preds)

    def _encode_new(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        df = df.copy()
        if self.raw.numeric_cols:
            for c in self.raw.numeric_cols:
                if c not in df.columns:
                    raise ValueError(f"Missing required numeric column: {c}")
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            X_num = df[self.raw.numeric_cols].to_numpy(dtype=np.float32)
        else:
            X_num = np.zeros((len(df), 0), dtype=np.float32)

        if self.raw.categorical_cols:
            X_cat = np.zeros((len(df), len(self.raw.categorical_cols)), dtype=np.int64)
            for j, c in enumerate(self.raw.categorical_cols):
                if c not in df.columns:
                    raise ValueError(f"Missing required categorical column: {c}")
                mapping = self.raw.cat_value_to_idx[j]
                s = df[c].fillna("__missing__").astype(str)
                X_cat[:, j] = s.map(mapping).fillna(0).astype(np.int64).to_numpy()
        else:
            X_cat = np.zeros((len(df), 0), dtype=np.int64)
        return X_num, X_cat

    @staticmethod
    def _print_spec(spec: ArchSpec) -> None:
        print(
            f"Proposed: arch={spec.architecture_type} hidden={spec.hidden_sizes} "
            f"act={spec.activation} dropout={spec.dropout} bn={spec.batch_norm}\n"
            f"          opt={spec.optimizer} lr={spec.learning_rate} wd={spec.weight_decay} "
            f"bs={spec.batch_size} epochs={spec.epochs} sched={spec.lr_scheduler}\n"
            f"          numeric_scaler={spec.numeric_scaler} cat_enc={spec.categorical_encoding}"
            f" emb_dim={spec.embedding_dim} target_std={spec.target_standardize}"
        )
        if spec.rationale:
            print(f"Rationale: {spec.rationale}")

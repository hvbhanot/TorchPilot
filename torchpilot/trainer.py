from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from torchpilot.data import Preprocessor, RawData, loaders_from_preprocessor
from torchpilot.metrics import classification_metrics, primary_metric_name, regression_metrics
from torchpilot.model import TabularNet, build_optimizer, build_scheduler
from torchpilot.schemas import ArchSpec


@dataclass
class TrainOutcome:
    train_loss_history: list[float] = field(default_factory=list)
    val_loss_history: list[float] = field(default_factory=list)
    val_metric_history: list[float] = field(default_factory=list)
    final_metrics: dict[str, float] = field(default_factory=dict)
    metric_name: str = ""
    epochs_run: int = 0
    best_state: dict = field(default_factory=dict)
    trajectory: str = ""

    @property
    def best_val_loss(self) -> float:
        return min(self.val_loss_history) if self.val_loss_history else float("nan")

    @property
    def primary_val_metric(self) -> float:
        return self.final_metrics.get(self.metric_name, float("nan"))


def _loss_fn(task: str) -> nn.Module:
    return nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()


@torch.no_grad()
def _eval(model: TabularNet, loader, task: str, loss_fn, dev, prep: Preprocessor):
    model.eval()
    total_loss, n = 0.0, 0
    y_true_chunks, y_pred_chunks, proba_chunks = [], [], []
    for x_num, x_cat, yb in loader:
        x_num, x_cat, yb = x_num.to(dev), x_cat.to(dev), yb.to(dev)
        out = model(x_num, x_cat)
        if task == "regression":
            out = out.squeeze(-1)
        loss = loss_fn(out, yb)
        total_loss += loss.item() * x_num.size(0)
        n += x_num.size(0)
        y_np = yb.detach().cpu().numpy()
        if task == "classification":
            proba = torch.softmax(out, dim=1).detach().cpu().numpy()
            proba_chunks.append(proba)
            y_pred_chunks.append(proba.argmax(axis=1))
            y_true_chunks.append(y_np)
        else:
            preds = out.detach().cpu().numpy()
            # Invert target standardization for metric reporting (loss stays in standardized space).
            y_true_chunks.append(prep.inverse_target(y_np))
            y_pred_chunks.append(prep.inverse_target(preds))

    avg_loss = total_loss / max(n, 1)
    y_true = np.concatenate(y_true_chunks)
    y_pred = np.concatenate(y_pred_chunks)
    y_proba = np.concatenate(proba_chunks) if proba_chunks else None
    return avg_loss, y_true, y_pred, y_proba


def _all_metrics(task: str, y_true, y_pred, y_proba, n_classes: int) -> dict[str, float]:
    if task == "classification":
        return classification_metrics(y_true, y_pred, y_proba, n_classes)
    return regression_metrics(y_true, y_pred)


def _trajectory_note(
    train_losses: list[float], val_losses: list[float], val_metrics: list[float], task: str
) -> str:
    """Diagnose the learning curve so the LLM can react meaningfully."""
    if len(val_losses) < 2:
        return "too few epochs to diagnose"

    notes: list[str] = []
    final_train, final_val = train_losses[-1], val_losses[-1]
    best_val = min(val_losses)
    best_epoch = val_losses.index(best_val) + 1

    if not np.isfinite(final_val) or final_val > 1e6:
        notes.append("loss diverged")

    gap = final_val - final_train
    if gap > 0.4 * max(final_train, 1e-6) and gap > 0.05:
        notes.append(f"overfitting (val-train gap {gap:.3f})")

    last_k = min(5, len(val_losses))
    recent = val_losses[-last_k:]
    if max(recent) - min(recent) < 0.01 * max(abs(np.mean(recent)), 1e-3):
        notes.append("val loss plateaued in last epochs")

    if best_epoch == len(val_losses) and len(val_losses) >= 5:
        notes.append("still improving at last epoch (try more epochs)")

    if not notes:
        # Underfit heuristic: both losses high relative to typical scale.
        if task == "classification" and val_metrics and val_metrics[-1] < 0.6:
            notes.append(f"underfitting (val accuracy {val_metrics[-1]:.2f})")
        else:
            notes.append("clean convergence")

    return "; ".join(notes)


def train(
    spec: ArchSpec,
    raw: RawData,
    patience: int = 10,
    device: str | None = None,
    verbose: bool = False,
) -> tuple[TabularNet, Preprocessor, TrainOutcome]:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    prep = Preprocessor(spec=spec).fit(raw)
    train_loader, val_loader = loaders_from_preprocessor(raw, prep, spec.batch_size)

    model = TabularNet(spec, raw.n_numeric, raw.cat_cardinalities, raw.n_outputs).to(dev)
    optim = build_optimizer(spec, model)
    sched = build_scheduler(spec, optim)
    loss_fn = _loss_fn(raw.task)
    metric_name = primary_metric_name(raw.task)

    best_val_loss = float("inf")
    best_state: dict = {}
    epochs_no_improve = 0
    outcome = TrainOutcome(metric_name=metric_name)

    for epoch in range(1, spec.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for x_num, x_cat, yb in train_loader:
            x_num, x_cat, yb = x_num.to(dev), x_cat.to(dev), yb.to(dev)
            optim.zero_grad()
            out = model(x_num, x_cat)
            if raw.task == "regression":
                out = out.squeeze(-1)
            loss = loss_fn(out, yb)
            loss.backward()
            optim.step()
            running += loss.item() * x_num.size(0)
            n += x_num.size(0)
        train_loss = running / max(n, 1)

        val_loss, y_true, y_pred, y_proba = _eval(model, val_loader, raw.task, loss_fn, dev, prep)
        epoch_metrics = _all_metrics(raw.task, y_true, y_pred, y_proba, raw.n_outputs)
        primary = epoch_metrics[metric_name]

        outcome.train_loss_history.append(train_loss)
        outcome.val_loss_history.append(val_loss)
        outcome.val_metric_history.append(primary)
        outcome.epochs_run = epoch

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            outcome.final_metrics = epoch_metrics
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if sched is not None:
            sched.step()

        if verbose:
            extras = " ".join(f"{k}={v:.4f}" for k, v in epoch_metrics.items() if k != metric_name)
            print(
                f"  epoch {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                f"val_{metric_name} {primary:.4f} | {extras}"
            )

        if epochs_no_improve >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    outcome.best_state = best_state
    outcome.trajectory = _trajectory_note(
        outcome.train_loss_history, outcome.val_loss_history, outcome.val_metric_history, raw.task
    )
    return model, prep, outcome

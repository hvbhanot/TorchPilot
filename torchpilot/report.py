"""Multi-page PDF report of an TorchPilot run."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from torchpilot.metrics import primary_better, primary_metric_name
from torchpilot.schemas import ArchSpec, TrialResult


def _text_page(pdf: PdfPages, title: str, body_lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.07, 0.94, title, fontsize=18, fontweight="bold")
    text = "\n".join(body_lines)
    fig.text(0.07, 0.90, text, fontsize=10, family="monospace", va="top", wrap=True)
    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def _summary_table_page(pdf: PdfPages, history: list[TrialResult], task: str) -> None:
    metric_keys = sorted({k for t in history for k in t.final_metrics.keys()})
    headers = ["Round", "Arch", "Hidden", "Opt", "LR", "Sched", "Drop", "Scaler", "CatEnc", "Epochs", "val_loss", *metric_keys]
    rows = []
    primary = primary_metric_name(task)
    best_idx = _best_index(history, task)
    for i, t in enumerate(history):
        row = [
            str(t.round),
            "res" if t.spec.architecture_type == "residual_mlp" else "mlp",
            "x".join(map(str, t.spec.hidden_sizes)),
            t.spec.optimizer,
            f"{t.spec.learning_rate:.0e}",
            t.spec.lr_scheduler,
            f"{t.spec.dropout:.2f}",
            t.spec.numeric_scaler,
            "emb" if t.spec.categorical_encoding == "embedding" else "1hot",
            str(t.epochs_run),
            f"{t.val_loss:.4f}",
        ]
        for k in metric_keys:
            v = t.final_metrics.get(k, float("nan"))
            row.append(f"{v:.4f}")
        if i == best_idx:
            row[0] = f"★ {row[0]}"
        rows.append(row)

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title(
        f"All Rounds — Summary (★ best by val {primary})",
        fontsize=14, fontweight="bold", loc="left", pad=20,
    )
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _round_page(pdf: PdfPages, trial: TrialResult, profile_task: str) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(
        f"Round {trial.round} — val_loss {trial.val_loss:.4f} | "
        f"val_{trial.metric_name} {trial.val_metric:.4f}",
        fontsize=14, fontweight="bold",
    )

    # Top-left: spec text
    ax_spec = fig.add_axes([0.05, 0.55, 0.42, 0.35])
    ax_spec.axis("off")
    ax_spec.set_title("Configuration", fontsize=11, loc="left", fontweight="bold")
    spec_lines = [
        f"architecture   : {trial.spec.architecture_type}",
        f"hidden_sizes   : {trial.spec.hidden_sizes}",
        f"activation     : {trial.spec.activation}",
        f"dropout        : {trial.spec.dropout}",
        f"batch_norm     : {trial.spec.batch_norm}",
        f"optimizer      : {trial.spec.optimizer}",
        f"learning_rate  : {trial.spec.learning_rate}",
        f"weight_decay   : {trial.spec.weight_decay}",
        f"batch_size     : {trial.spec.batch_size}",
        f"max_epochs     : {trial.spec.epochs}",
        f"epochs_run     : {trial.epochs_run}",
        f"lr_scheduler   : {trial.spec.lr_scheduler}",
        f"numeric_scaler : {trial.spec.numeric_scaler}",
        f"cat_encoding   : {trial.spec.categorical_encoding}",
        f"embedding_dim  : {trial.spec.embedding_dim}",
        f"target_std     : {trial.spec.target_standardize}",
    ]
    ax_spec.text(0, 1, "\n".join(spec_lines), fontsize=9, family="monospace", va="top")

    # Top-right: final metrics table
    ax_metrics = fig.add_axes([0.55, 0.55, 0.40, 0.35])
    ax_metrics.axis("off")
    ax_metrics.set_title("Final validation metrics", fontsize=11, loc="left", fontweight="bold")
    if trial.final_metrics:
        rows = [[k, f"{v:.4f}"] for k, v in trial.final_metrics.items()]
        tbl = ax_metrics.table(cellText=rows, colLabels=["metric", "value"], loc="upper left", cellLoc="left")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.4)

    # Bottom-left: loss curves
    ax_loss = fig.add_axes([0.08, 0.08, 0.40, 0.38])
    epochs = range(1, len(trial.train_loss_history) + 1)
    ax_loss.plot(epochs, trial.train_loss_history, label="train")
    ax_loss.plot(epochs, trial.val_loss_history, label="val")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Loss curves", fontsize=11)
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # Bottom-right: primary val metric curve
    ax_m = fig.add_axes([0.56, 0.08, 0.40, 0.38])
    ax_m.plot(epochs, trial.val_metric_history, color="C2")
    ax_m.set_xlabel("epoch")
    ax_m.set_ylabel(f"val {trial.metric_name}")
    ax_m.set_title(f"Validation {trial.metric_name} per epoch", fontsize=11)
    ax_m.grid(True, alpha=0.3)

    # Rationale + trajectory strip
    info_lines = []
    if trial.trajectory:
        info_lines.append(f"Trajectory: {trial.trajectory}")
    if trial.spec.rationale:
        info_lines.append(f"LLM rationale: {trial.spec.rationale}")
    if info_lines:
        fig.text(0.05, 0.50, "\n".join(info_lines), fontsize=9, style="italic", wrap=True)

    pdf.savefig(fig)
    plt.close(fig)


def _best_index(history: list[TrialResult], task: str) -> int:
    primary = primary_metric_name(task)
    best_i, best_v = 0, history[0].final_metrics.get(primary, float("nan"))
    for i, t in enumerate(history[1:], start=1):
        v = t.final_metrics.get(primary, float("nan"))
        if primary_better(task, v, best_v):
            best_v, best_i = v, i
    return best_i


def _format_profile(profile: dict) -> list[str]:
    lines = [
        f"Task                : {profile['task']}",
        f"Rows                : {profile['n_rows']}",
        f"Numeric columns     : {profile['n_numeric_columns']}",
        f"Categorical columns : {profile['n_categorical_columns']}",
        f"Target              : {profile['target']['name']}",
    ]
    cards = profile.get("categorical_cardinalities", {})
    if cards:
        lines.append("Categorical cardinalities :")
        for k, v in cards.items():
            lines.append(f"    {k}: {v}")
    target = profile["target"]
    if profile["task"] == "classification":
        lines.append(f"Number of classes          : {target.get('n_classes')}")
        balance = target.get("class_balance", {})
        if balance:
            lines.append("Class balance              :")
            for k, v in balance.items():
                lines.append(f"    {k}: {v:.3f}")
    else:
        lines.append(
            f"Target stats               : min={target['min']:.3f} max={target['max']:.3f} "
            f"mean={target['mean']:.3f} std={target['std']:.3f}"
        )
    return lines


def write_report(
    output_path: str | Path,
    profile: dict,
    history: list[TrialResult],
    best_spec: ArchSpec,
    best_result: TrialResult,
    csv_path: str,
    llm_model: str,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    task = profile["task"]
    primary = primary_metric_name(task)

    with PdfPages(output) as pdf:
        # Cover
        cover_lines = [
            f"Generated  : {datetime.now().isoformat(timespec='seconds')}",
            f"CSV        : {csv_path}",
            f"LLM model  : {llm_model}",
            f"Rounds run : {len(history)}",
            "",
            "Dataset",
            "-------",
            *_format_profile(profile),
            "",
            "Best round",
            "----------",
            f"Round        : {best_result.round}",
            f"val_loss     : {best_result.val_loss:.4f}",
            f"val_{primary:<8} : {best_result.val_metric:.4f}",
            f"Architecture : hidden={best_spec.hidden_sizes} "
            f"act={best_spec.activation} dropout={best_spec.dropout} bn={best_spec.batch_norm}",
            f"Optim        : {best_spec.optimizer} lr={best_spec.learning_rate} "
            f"wd={best_spec.weight_decay} bs={best_spec.batch_size}",
        ]
        _text_page(pdf, "TorchPilot Run Report", cover_lines)

        # Summary across rounds
        if history:
            _summary_table_page(pdf, history, task)

        # Per-round pages
        for trial in history:
            _round_page(pdf, trial, task)

    return output

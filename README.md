# TorchPilot

**LLM-driven AutoML for tabular PyTorch models.** Point it at a CSV, pick `classification` or
`regression`, and an Ollama Cloud model iteratively proposes architectures, hyperparameters,
*and preprocessing* — TorchPilot trains each proposal, feeds the metrics + a learning-curve
diagnostic back to the LLM, and the LLM revises across rounds. Best model wins, and you
get a PDF report.

```python
from torchpilot import TorchPilot
report = TorchPilot("data.csv", target="label", task="classification").fit(n_rounds=5, report_path="run.pdf")
```

---

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [CSV requirements](#csv-requirements)
- [Public API](#public-api)
- [What the LLM controls each round](#what-the-llm-controls-each-round)
- [Trajectory diagnostics](#trajectory-diagnostics)
- [Metrics tracked](#metrics-tracked)
- [PDF report](#pdf-report)
- [Examples](#examples)
- [Tips for getting good runs](#tips-for-getting-good-runs)
- [Troubleshooting](#troubleshooting)
- [Saving and loading the trained model](#saving-and-loading-the-trained-model)
- [Project layout](#project-layout)
- [Development](#development)

---

## Installation

### From local checkout (development)

```bash
git clone <this-repo>
cd TorchPilot
pip install -e .
```

`-e` (editable) means your local edits are picked up without reinstall — best while
hacking on the library.

### From a GitHub clone (regular user)

```bash
pip install git+https://github.com/hvbhanot/TorchPilot.git
```

### Ollama Cloud key

You need a key from [https://ollama.com](https://ollama.com). Export it before running:

```bash
export OLLAMA_API_KEY=sk-...
```

Either set it in your shell, your `.env`, or pass `api_key=...` directly to `TorchPilot(...)`.

### GPU (optional)

TorchPilot uses whatever PyTorch picks. If you have CUDA installed, pass `device="cuda"`;
otherwise it falls back to CPU automatically (which is fine for small tabular data).

---

## Quick start

```python
from torchpilot import TorchPilot

auto = TorchPilot(
    csv_path="data.csv",
    target="label",
    task="classification",              # or "regression"
    llm_model="gpt-oss:120b-cloud",     # any Ollama Cloud model
)
report = auto.fit(n_rounds=5, report_path="run.pdf")

print(report.best_spec)                       # ArchSpec the LLM converged on
print(report.best_result.final_metrics)       # {'accuracy': 0.93, 'f1': 0.92, 'auroc': 0.97, ...}

# Reuse on new data
import pandas as pd
preds = auto.predict(pd.read_csv("new_data.csv"))
```

### No-LLM smoke test (verifies install)

```bash
python -c "
import numpy as np, pandas as pd, tempfile
np.random.seed(0)
n = 300
df = pd.DataFrame({'x1': np.random.randn(n), 'x2': np.random.randn(n),
                   'y':  (np.random.randn(n) > 0).astype(int)})
csv = tempfile.mkstemp(suffix='.csv')[1]; df.to_csv(csv, index=False)

from torchpilot.data import load_raw
from torchpilot.schemas import ArchSpec
from torchpilot.trainer import train

raw = load_raw(csv, target='y', task='classification')
model, prep, out = train(ArchSpec(hidden_sizes=[16,8], epochs=10), raw, verbose=True)
print('final:', out.final_metrics, 'trajectory:', out.trajectory)
"
```

If that prints metrics, your install is good and only the LLM call is missing.

---

## CSV requirements

- A CSV that pandas can read (UTF-8, `,` delimiter, header row).
- One column is the target (passed via `target=...`).
- Every other column is treated as a feature.
- Numeric columns (`int`, `float`, `bool`) → median-imputed + scaled.
- Anything else (`object`, `string`, `category`) → categorical, integer-encoded, then
  one-hot or embedded depending on what the LLM picks.
- Missing target rows are dropped silently.
- No special escaping/quoting requirements beyond pandas defaults.

Recommended: have at least ~50 rows per class for classification, and validate that the
target column type matches the task you pass (numeric → regression, anything → classification
is also fine since labels get encoded).

---

## Public API

### `TorchPilot(csv_path, target, task, ...)`

| arg | type | default | meaning |
| --- | --- | --- | --- |
| `csv_path` | `str` | required | path to your CSV |
| `target` | `str` | required | name of the target column |
| `task` | `"classification"` \| `"regression"` | required | which loss / metrics to use |
| `llm_model` | `str` | `"gpt-oss:120b-cloud"` | any model available on Ollama Cloud |
| `api_key` | `str \| None` | `None` (reads `OLLAMA_API_KEY`) | overrides env var |
| `val_size` | `float` | `0.2` | fraction held out for validation |
| `random_state` | `int` | `42` | reproducible splits |
| `device` | `str \| None` | auto | `"cpu"`, `"cuda"`, `"mps"`, or `None` |

### `TorchPilot.fit(n_rounds=5, patience=10, verbose=True, report_path=None) -> FitReport`

Runs the iterative LLM ↔ train loop. Returns a `FitReport`:

```python
@dataclass
class FitReport:
    best_spec: ArchSpec           # the winning configuration
    best_result: TrialResult      # winning trial (metrics + curves)
    history: list[TrialResult]    # every round, in order
    report_path: Path | None      # PDF path if you passed report_path=
```

- `patience` — early-stopping patience on val loss (per training run, not per round).
- `report_path` — if set, writes a multi-page PDF with the full run history.

### `TorchPilot.predict(df: pd.DataFrame) -> np.ndarray`

Runs the best model on new rows. Same feature columns must be present (extra columns are
ignored). Categorical values not seen during training are routed to a reserved OOV slot.
Returns:
- Classification: the original label values (decoded via the stored `LabelEncoder`).
- Regression: predictions in the original scale (target standardization is inverted if
  the best spec used it).

### `TorchPilot.save(path) -> Path` / `TorchPilot.load(path) -> TorchPilot`

Persist the best model + spec + preprocessor + raw metadata to a single `.pt` file, and
reconstruct a predict-ready instance later (in a different process, on a different machine,
without an LLM key). See [Saving and loading the trained model](#saving-and-loading-the-trained-model).

### `TorchPilot.write_report(output_path) -> Path`

Write a PDF report after `fit()`. Equivalent to passing `report_path=` to `fit()`.

### `ArchSpec` (Pydantic model)

What the LLM produces each round. See [What the LLM controls each round](#what-the-llm-controls-each-round).
Inspect it like any Pydantic model:

```python
report.best_spec.model_dump()             # → dict
report.best_spec.model_dump_json(indent=2)  # → pretty JSON
```

### `TrialResult` (Pydantic model)

Carries per-round results: `spec`, `train_loss`, `val_loss`, `val_metric`, `metric_name`,
`epochs_run`, `final_metrics` (all metrics), `train_loss_history` / `val_loss_history` /
`val_metric_history` (per-epoch curves), and `trajectory` (the diagnostic string).

---

## What the LLM controls each round

The LLM emits an `ArchSpec` per round, conforming to a JSON schema (enforced via Ollama's
structured-output `format=...` parameter, with a one-shot repair attempt on validation
failure).

**Architecture**
- `architecture_type` — `mlp` (plain) or `residual_mlp` (pre-activation residual blocks
  with optional projection when in/out dims differ)
- `hidden_sizes` — list of layer widths, e.g. `[128, 64]`
- `activation` — `relu` / `gelu` / `tanh` / `leaky_relu`
- `dropout` (0 – 0.7), `batch_norm` (bool)

**Optimization**
- `optimizer` — `adam` / `adamw` / `sgd` (SGD uses momentum 0.9)
- `learning_rate`, `weight_decay`, `batch_size`, `epochs`
- `lr_scheduler` — `none` / `cosine` (annealing across `epochs`) / `step` (γ=0.5 every `epochs/3`)

**Preprocessing** (per round — *not fixed*)
- `numeric_scaler` — `standard` / `minmax` / `robust` / `none`
- `categorical_encoding` — `onehot` (fast, fine for low cardinality) or `embedding`
  (learned, preferred when any column has cardinality > ~20)
- `embedding_dim` — width of learned embedding vectors (applies to all cat columns)
- `target_standardize` — regression only; standardize y for training, invert for metrics
  and `predict()`. Strongly recommended when target magnitude is far from zero or unit-std.

All preprocessing is fit on TRAIN only (no leakage into val), per round.

---

## Trajectory diagnostics

After every round, TorchPilot summarizes the learning curves into a short diagnostic string
and sends it back to the LLM along with the metrics. Possible notes (any may be combined):

| Diagnostic | Heuristic |
| --- | --- |
| `"overfitting (val-train gap X)"` | val_loss − train_loss > 40% of train_loss |
| `"val loss plateaued in last epochs"` | last 5 epochs' val_loss span < 1% of mean |
| `"still improving at last epoch (try more epochs)"` | best val_loss is the final epoch |
| `"underfitting (val accuracy X)"` | classification primary metric < 0.6 |
| `"loss diverged"` | final val_loss non-finite or > 1e6 |
| `"clean convergence"` | none of the above triggered |

The system prompt explicitly instructs the LLM how to react to each (more dropout for
overfitting, larger nets for underfitting, cosine schedule for plateaus, lower LR for
divergence, etc).

---

## Metrics tracked

**Classification** — accuracy, precision, recall, F1, AUROC.
- Binary → `average="binary"`, AUROC on positive-class probabilities.
- Multiclass → macro averaging, AUROC via one-vs-rest macro.
- `zero_division=0` so empty-class predictions don't crash.

**Regression** — RMSE, MAE, R².

Primary metric (used for "best round" selection): **accuracy** for classification,
**RMSE** for regression. The LLM still sees all of them.

---

## PDF report

If you pass `report_path="run.pdf"` (or call `auto.write_report(...)`), TorchPilot writes a
multi-page PDF containing:

1. **Cover** — timestamp, CSV path, LLM model, rounds, full dataset profile (rows,
   numeric/categorical counts, categorical cardinalities, target stats / class balance),
   and the best round's headline result.
2. **Summary table** — every round in one table: architecture, hidden sizes, optimizer,
   LR, scheduler, dropout, scaler, encoding, epochs run, val_loss, and every final metric.
   The best round is starred (★).
3. **Per-round pages** — one page each: full `ArchSpec`, all final validation metrics in a
   table, train/val loss curves, validation-metric curve per epoch, plus the LLM's
   rationale and the trajectory diagnostic.

---

## Examples

### Classification (Iris)

See `examples/iris_example.py`. Run it directly:

```bash
python examples/iris_example.py
```

### Regression (synthetic linear data)

```python
import numpy as np, pandas as pd
from torchpilot import TorchPilot

np.random.seed(0)
n = 1000
df = pd.DataFrame({
    "x1": np.random.randn(n),
    "x2": np.random.randn(n),
    "category": np.random.choice(["a","b","c","d"], size=n),
    "y": (50 + 30 * np.random.randn(n)),  # mean ~50, std ~30 → target_standardize matters
})
df.to_csv("synth.csv", index=False)

auto = TorchPilot("synth.csv", target="y", task="regression")
report = auto.fit(n_rounds=4, report_path="synth_run.pdf")
print("Best RMSE:", report.best_result.final_metrics["rmse"])
print("R²:       ", report.best_result.final_metrics["r2"])
```

### Inspecting what the LLM chose

```python
report = auto.fit(n_rounds=5)
for trial in report.history:
    print(f"Round {trial.round}: {trial.spec.architecture_type} {trial.spec.hidden_sizes} "
          f"lr={trial.spec.learning_rate} → val_loss={trial.val_loss:.4f} "
          f"({trial.trajectory})")
```

---

## Tips for getting good runs

- **More rounds → better results, more $$$.** Start with 3 rounds to gauge cost, then
  scale up. Each round = one LLM call + one full training run.
- **Small datasets (<1k rows)**: 3–5 rounds usually plateaus. Lower `patience` (e.g. `5`)
  to spend less time per round.
- **High-cardinality categoricals**: trust the LLM to pick `embedding` once it sees the
  profile, but you can short-circuit by setting a custom `random_state` and re-running.
- **Class imbalance**: TorchPilot reports per-class F1/precision/recall, but doesn't
  *resample* — apply your own sampling/weighting upstream if needed.
- **Reproducibility**: pass a fixed `random_state` and ask the LLM via a low temperature
  (currently 0.4 — patch `llm.py` if you want it lower).

---

## Troubleshooting

**`RuntimeError: Ollama Cloud API key required`**
Set `OLLAMA_API_KEY` or pass `api_key="..."` to `TorchPilot(...)`.

**`ValidationError: ...` on the LLM response**
TorchPilot automatically does one repair attempt; if it still fails the request is
malformed. Try a different `llm_model` (e.g. `deepseek-v3.1:671b-cloud`).

**Tiny dataset, accuracy ~chance**
Real signal may be missing, or the val split happens to be unfavorable. Try
`val_size=0.3`, increase `n_rounds`, or check class balance.

**CUDA OOM**
Pass `device="cpu"` — tabular MLPs on CPU are usually fine for <100k rows.

**Categorical column with millions of unique values (e.g. a user ID)**
Drop it or hash it. TorchPilot will treat it as cardinality-N and either one-hot it (bad)
or embed it (still memory-heavy). It's not a real feature.

---

## Saving and loading the trained model

The best model from a run can be persisted to disk and reloaded later — including the
preprocessor (scaler / target standardization stats / categorical mappings) so `predict()`
works on fresh dataframes without needing the original CSV.

**Auto-save while fitting** (pass `save_path` alongside `report_path`):

```python
report = pilot.fit(n_rounds=5,
                   report_path="run.pdf",
                   save_path="best_model.pt")
print(report.model_path)  # → Path('best_model.pt')
```

**Manual save** after fitting:

```python
pilot.save("best_model.pt")
```

**Load and predict** in a fresh process — no LLM key needed:

```python
from torchpilot import TorchPilot
pilot = TorchPilot.load("best_model.pt")
preds = pilot.predict(new_df)
```

The checkpoint is a single `torch.save`-format file containing:
- Model `state_dict` (CPU tensors)
- The winning `ArchSpec`
- The fitted `Preprocessor` (numeric scaler + target stats)
- Raw metadata (column names, categorical mappings, label encoder, `n_outputs`)
- Dataset profile + best metrics for reference

A loaded instance can `predict()` but cannot `fit()` (no architect attached — retraining
requires constructing a fresh `TorchPilot(csv_path, target, task)`).

---

## Project layout

```
TorchPilot/
├── pyproject.toml
├── README.md
├── torchpilot/
│   ├── __init__.py          # public exports: TorchPilot, ArchSpec, TrialResult
│   ├── schemas.py           # Pydantic models (ArchSpec, TrialResult)
│   ├── data.py              # load_raw + Preprocessor (per-spec fit-on-train)
│   ├── model.py             # TabularNet (mlp / residual_mlp, onehot / embedding)
│   ├── metrics.py           # classification + regression metric funcs
│   ├── trainer.py           # training loop, early stopping, trajectory diagnostic
│   ├── llm.py               # Ollama Cloud client + system prompt + JSON-schema chat
│   ├── autotrainer.py       # TorchPilot orchestrator: fit / predict / write_report
│   └── report.py            # matplotlib PdfPages report writer
└── examples/
    └── iris_example.py
```

---

## Development

```bash
git clone https://github.com/hvbhanot/TorchPilot
cd TorchPilot
pip install -e ".[dev]"     # editable install + pytest
```

To run a syntax check on every file:

```bash
python -m py_compile torchpilot/*.py examples/*.py
```

Contributions: open an issue describing the change first if it touches the LLM prompt,
the `ArchSpec` schema, or the data layout — those affect the LLM contract and need
thinking through.

---

## License

MIT.

from typing import Literal
from pydantic import BaseModel, Field, field_validator


def _clamp(v, lo, hi, default):
    try:
        x = type(default)(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, x))


class ArchSpec(BaseModel):
    # Architecture
    architecture_type: Literal["mlp", "residual_mlp"] = "mlp"
    hidden_sizes: list[int] = Field(..., description="Width of each hidden layer (block), in order.")
    activation: Literal["relu", "gelu", "tanh", "leaky_relu"] = "relu"
    dropout: float = Field(0.0, description="Dropout probability, clamped to [0, 0.7].")
    batch_norm: bool = False

    # Optimization
    optimizer: Literal["adam", "adamw", "sgd"] = "adam"
    learning_rate: float = Field(1e-3, description="LR, clamped to (1e-6, 1].")
    weight_decay: float = Field(0.0, description="Weight decay, clamped to [0, 1].")
    batch_size: int = Field(32, description="Batch size, clamped to [1, 4096].")
    epochs: int = Field(50, description="Max epochs, clamped to [1, 500].")
    lr_scheduler: Literal["none", "cosine", "step"] = "none"

    # Preprocessing (chosen by the LLM)
    numeric_scaler: Literal["standard", "minmax", "robust", "none"] = "standard"
    categorical_encoding: Literal["onehot", "embedding"] = "onehot"
    embedding_dim: int = Field(
        8,
        description=(
            "Embedding dimensionality per categorical column when encoding=embedding. "
            "Unused when encoding=onehot or there are no categorical columns; clamped to [1, 64]."
        ),
    )
    target_standardize: bool = Field(
        False,
        description="Regression only: standardize y during training, invert for metrics/predictions.",
    )

    rationale: str = Field("", description="Short justification for this configuration.")

    # Coerce-on-validate: the LLM occasionally emits out-of-range values (e.g. embedding_dim=0
    # when there are no categoricals). We clamp instead of rejecting so a single round can't
    # crash the entire fit().
    @field_validator("dropout", mode="before")
    @classmethod
    def _v_dropout(cls, v):
        return _clamp(v, 0.0, 0.7, 0.0)

    @field_validator("learning_rate", mode="before")
    @classmethod
    def _v_lr(cls, v):
        return _clamp(v, 1e-6, 1.0, 1e-3)

    @field_validator("weight_decay", mode="before")
    @classmethod
    def _v_wd(cls, v):
        return _clamp(v, 0.0, 1.0, 0.0)

    @field_validator("batch_size", mode="before")
    @classmethod
    def _v_bs(cls, v):
        return _clamp(v, 1, 4096, 32)

    @field_validator("epochs", mode="before")
    @classmethod
    def _v_epochs(cls, v):
        return _clamp(v, 1, 500, 50)

    @field_validator("embedding_dim", mode="before")
    @classmethod
    def _v_embed_dim(cls, v):
        return _clamp(v, 1, 64, 8)

    @field_validator("hidden_sizes", mode="before")
    @classmethod
    def _v_hidden(cls, v):
        if not isinstance(v, list) or not v:
            return [64, 32]
        out = []
        for h in v:
            try:
                hi = int(h)
                if hi > 0:
                    out.append(min(hi, 4096))
            except (TypeError, ValueError):
                continue
        return out or [64, 32]


class TrialResult(BaseModel):
    round: int
    spec: ArchSpec
    train_loss: float
    val_loss: float
    val_metric: float
    metric_name: str
    epochs_run: int
    final_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="All metrics computed on the best val state (accuracy, f1, auroc, etc).",
    )
    train_loss_history: list[float] = Field(default_factory=list)
    val_loss_history: list[float] = Field(default_factory=list)
    val_metric_history: list[float] = Field(default_factory=list)
    trajectory: str = Field(
        "",
        description="Diagnostic note about the learning curve (overfit / underfit / plateau / improving / diverging).",
    )
    notes: str = ""

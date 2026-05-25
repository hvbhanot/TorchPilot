import json
import os
from typing import Any

from ollama import Client
from pydantic import ValidationError

from torchpilot.schemas import ArchSpec, TrialResult


SYSTEM_PROMPT = """You are an expert ML engineer designing tabular neural networks.
Each round you receive a dataset profile and the history of previous trials, including a
short "trajectory" diagnostic per trial (e.g. "overfitting", "still improving", "plateaued",
"underfitting", "diverged"). React to that diagnostic deliberately.

You control:
- architecture_type: "mlp" (plain) or "residual_mlp" (deeper, easier to train)
- hidden_sizes, activation, dropout, batch_norm
- optimizer (adam/adamw/sgd), learning_rate, weight_decay, batch_size, epochs
- lr_scheduler: "none" / "cosine" / "step"
- numeric_scaler: "standard" / "minmax" / "robust" / "none"
- categorical_encoding: "onehot" (good for low cardinality) or "embedding"
  (preferred when any categorical column has cardinality > ~20; pick embedding_dim accordingly,
   roughly min(50, max(2, cardinality // 2)) — but a single embedding_dim applies to all cats)
- target_standardize: regression only — strongly recommended when target std is large or off-zero

Rules of thumb:
- Small data (<1k rows): shallow (1-2 layers), strong dropout (0.2-0.4), small batch (8-32).
- Medium (1k-50k): 2-3 layers, batch 32-128.
- Large (>50k): residual_mlp helps; batch_norm + cosine schedule often wins.
- Overfitting trajectory -> more dropout, more weight_decay, smaller net, or early-stop sooner.
- Underfitting -> larger hidden_sizes, residual_mlp, more epochs.
- Plateau -> try cosine schedule, different optimizer, or change activation.
- Diverged -> drop learning_rate by 10x, consider standard scaler if you used minmax/none.
- Don't repeat a configuration that already underperformed; change at least one major lever.

Respond ONLY with JSON matching the provided schema. Include a brief `rationale` referencing
what you learned from prior trials (if any)."""


class LLMArchitect:
    def __init__(
        self,
        model: str = "gpt-oss:120b-cloud",
        api_key: str | None = None,
        host: str = "https://ollama.com",
    ):
        key = api_key or os.environ.get("OLLAMA_API_KEY")
        if not key:
            raise RuntimeError(
                "Ollama Cloud API key required. Pass api_key=... or set OLLAMA_API_KEY."
            )
        self.model = model
        self.client = Client(host=host, headers={"Authorization": f"Bearer {key}"})

    def propose(self, profile: dict[str, Any], history: list[TrialResult]) -> ArchSpec:
        user_msg = {
            "dataset_profile": profile,
            "previous_trials": [self._summarize_trial(t) for t in history],
            "round": len(history) + 1,
        }
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_msg, default=str)},
            ],
            format=ArchSpec.model_json_schema(),
            options={"temperature": 0.4},
        )
        raw = response["message"]["content"]
        try:
            return ArchSpec.model_validate_json(raw)
        except ValidationError as e:
            repair = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(user_msg, default=str)},
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": f"Your JSON failed validation: {e}. Return corrected JSON only.",
                    },
                ],
                format=ArchSpec.model_json_schema(),
                options={"temperature": 0.0},
            )
            return ArchSpec.model_validate_json(repair["message"]["content"])

    @staticmethod
    def _summarize_trial(t: TrialResult) -> dict:
        """Compact view — strip per-epoch curves, keep diagnostics + final metrics."""
        return {
            "round": t.round,
            "spec": t.spec.model_dump(),
            "train_loss": round(t.train_loss, 5),
            "val_loss": round(t.val_loss, 5),
            "primary_metric": {t.metric_name: round(t.val_metric, 5)},
            "all_final_metrics": {k: round(v, 5) for k, v in t.final_metrics.items()},
            "epochs_run": t.epochs_run,
            "trajectory": t.trajectory,
        }

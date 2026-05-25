"""Quick end-to-end demo on the Iris dataset.

Set OLLAMA_API_KEY in your env, then:
    python examples/iris_example.py
"""
from pathlib import Path

from sklearn.datasets import load_iris

from torchpilot import TorchPilot


def main() -> None:
    iris = load_iris(as_frame=True)
    df = iris.frame.rename(columns={"target": "species"})
    here = Path(__file__).parent
    csv_path = here / "iris.csv"
    df.to_csv(csv_path, index=False)

    pilot = TorchPilot(
        csv_path=str(csv_path),
        target="species",
        task="classification",
        llm_model="gpt-oss:120b-cloud",
    )
    report = pilot.fit(
        n_rounds=3,
        report_path=here / "iris_report.pdf",
        save_path=here / "iris_model.pt",
    )

    print("\n=== Best configuration ===")
    print(report.best_spec.model_dump_json(indent=2))
    print(f"\nBest val accuracy: {report.best_result.val_metric:.4f}")
    print(f"PDF report : {report.report_path}")
    print(f"Saved model: {report.model_path}")

    preds = pilot.predict(df.drop(columns=["species"]).head())
    print(f"In-process predictions   : {preds}")

    # Roundtrip: load the saved model in a fresh instance and re-predict.
    reloaded = TorchPilot.load(report.model_path)
    preds_loaded = reloaded.predict(df.drop(columns=["species"]).head())
    print(f"Predictions after load() : {preds_loaded}")


if __name__ == "__main__":
    main()

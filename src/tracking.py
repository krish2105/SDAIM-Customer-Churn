"""MLflow experiment tracking and model registry (H2-1).

Closes gap G5. Version 1.0.0 stored its results in a CSV, which meant a second
model version could not be compared to the first except by reading a file.

**Be precise about what this adds.** The project already achieves
reproducibility through fixed seeds and pinned versions. MLflow does not add
reproducibility — claiming that would be wrong. What it adds is
**comparability across runs** and **rollback to a previous version**, neither of
which a seed can give you.

Scope is deliberately limited to the local file backend. A tracking server, a
database or a cloud backend would add real operational cost to a project whose
demonstration value is identical without them.

MLflow is a **development** dependency only. It is never imported by the
deployed application, and the runtime image does not contain it — a test asserts
that separation.

CLI::

    python -m src.tracking --log-current
    python -m src.tracking --ui-hint
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src import config

EXPERIMENT_NAME = "customer-churn-intelligence"
REGISTERED_MODEL_NAME = "churn-retention-classifier"

#: Local SQLite backend. MLflow 3.x retired the plain filesystem store, and
#: opting out of that deprecation via MLFLOW_ALLOW_FILE_STORE would be borrowing
#: trouble. SQLite is still a single local file with no server to run, so the
#: "no infrastructure" scope limit is unaffected.
TRACKING_DIR: Path = config.PROJECT_ROOT / "mlruns"
TRACKING_DB: Path = TRACKING_DIR / "mlflow.db"
TRACKING_URI: str = f"sqlite:///{TRACKING_DB}"
ARTIFACT_URI: str = (TRACKING_DIR / "artifacts").as_uri()


def _require_mlflow():
    """Import MLflow with an actionable message when it is absent."""
    try:
        import mlflow  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "MLflow is not installed. It is a development dependency:\n"
            "    pip install -r requirements-dev.txt\n"
            "It is deliberately excluded from the deployment image."
        ) from exc
    return mlflow


def configure() -> Any:
    """Point MLflow at the local SQLite backend and select the experiment."""
    mlflow = _require_mlflow()
    (TRACKING_DIR / "artifacts").mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACT_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow


def log_training_run(
    results: dict[str, Any],
    *,
    register: bool = True,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Log one completed training run, including every candidate compared.

    Each candidate model gets a nested run so the comparison that drove selection
    is itself recorded — not just the winner. Recording only the selected model
    would lose the evidence for *why* it was selected.

    Args:
        results: The dictionary returned by ``src.train.run_training``.
        register: Whether to register the selected model in the registry.
        run_name: Optional label for the parent run.
    """
    mlflow = configure()
    metadata = results["metadata"]
    selected = results["selected_model"]

    logged: dict[str, Any] = {"experiment": EXPERIMENT_NAME, "candidates": {}}

    with mlflow.start_run(run_name=run_name or f"train-v{metadata['model_version']}") as parent:
        logged["run_id"] = parent.info.run_id

        mlflow.log_params(
            {
                "model_version": metadata["model_version"],
                "selected_model": selected,
                "random_state": metadata["random_state"],
                "test_size": metadata["test_size"],
                "cv_folds": metadata["cv_folds"],
                "decision_threshold": metadata["decision_threshold"],
                "train_rows": metadata["train_rows"],
                "test_rows": metadata["test_rows"],
            }
        )
        # The dataset SHA is the link between a run and the exact data it used.
        # Without it, a logged metric cannot be attributed to a data revision.
        mlflow.set_tags(
            {
                "dataset_git_blob_sha": metadata["dataset_git_blob_sha"],
                "dataset_rows": metadata["dataset_rows"],
                "selection_rule": metadata["selection_rule"],
                "selection_justification": metadata["selection_justification"],
                "data_notice": metadata["data_notice"],
                "governance_notice": metadata["governance_notice"],
                "scikit_learn": metadata["environment"]["scikit_learn"],
                "python": metadata["environment"]["python"],
            }
        )
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in metadata["metrics"].items()})
        mlflow.log_metrics(
            {k: float(v) for k, v in metadata["cross_validated_metrics"].items()}
        )
        mlflow.log_metrics(
            {f"confusion_{k}": float(v) for k, v in metadata["confusion_matrix"].items()}
        )

        for artifact in (
            config.METADATA_PATH,
            config.FEATURE_SCHEMA_PATH,
            config.MODEL_CARD_PATH,
            config.REPORTS_DIR / "model_comparison.csv",
            config.REPORTS_DIR / "executive_model_summary.md",
        ):
            if artifact.is_file():
                mlflow.log_artifact(str(artifact))

        for figure in sorted(config.FIGURES_DIR.glob("*.png")):
            mlflow.log_artifact(str(figure), artifact_path="figures")

        # Nested runs: one per candidate, so the comparison is inspectable.
        for row in results["comparison_rows"]:
            with mlflow.start_run(run_name=row["model"], nested=True) as child:
                mlflow.log_params({"model": row["model"], "role": row["role"]})
                mlflow.log_metrics(
                    {
                        key: float(value)
                        for key, value in row.items()
                        if isinstance(value, (int, float)) and not isinstance(value, bool)
                    }
                )
                mlflow.set_tag("selected", str(row["selected"]))
                logged["candidates"][row["model"]] = child.info.run_id

        if register and config.MODEL_PATH.is_file():
            import joblib  # noqa: PLC0415

            pipeline = joblib.load(config.MODEL_PATH)
            # MLflow 3.x serialises with skops, which refuses unlisted types by
            # default. numpy.dtype is what the ColumnTransformer records for its
            # column dtypes — a plain data descriptor with no executable payload,
            # so trusting it explicitly is safe. Declaring the single type is
            # preferable to falling back to pickle, which would trust everything.
            info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                registered_model_name=REGISTERED_MODEL_NAME if register else None,
                skops_trusted_types=["numpy.dtype"],
            )
            logged["model_uri"] = info.model_uri

            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
            if versions:
                newest = max(versions, key=lambda v: int(v.version))
                logged["registered_version"] = newest.version
                # Aliases replaced stage transitions in MLflow 2.9+. The alias
                # is what a rollback re-points, so it is the meaningful handle.
                client.set_registered_model_alias(
                    REGISTERED_MODEL_NAME, "production", newest.version
                )
                client.update_model_version(
                    name=REGISTERED_MODEL_NAME,
                    version=newest.version,
                    description=(
                        f"{selected} v{metadata['model_version']} — "
                        f"test ROC-AUC {metadata['metrics']['roc_auc']:.4f}, "
                        f"recall {metadata['metrics']['recall']:.4f}. "
                        f"Dataset blob {metadata['dataset_git_blob_sha'][:12]}."
                    ),
                )

    return logged


def list_runs() -> list[dict[str, Any]]:
    """Summarise logged parent runs, newest first."""
    mlflow = configure()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return []

    frame = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
    )
    if frame.empty:
        return []

    # Parent runs only — nested candidate runs carry a parent tag.
    parent_column = "tags.mlflow.parentRunId"
    if parent_column in frame.columns:
        frame = frame[frame[parent_column].isna()]

    rows: list[dict[str, Any]] = []
    for _, run in frame.iterrows():
        rows.append(
            {
                "run_id": run["run_id"],
                "run_name": run.get("tags.mlflow.runName", ""),
                "started": str(run["start_time"]),
                "model": run.get("params.selected_model", ""),
                "version": run.get("params.model_version", ""),
                "test_roc_auc": run.get("metrics.test_roc_auc"),
                "test_recall": run.get("metrics.test_recall"),
                "test_f1": run.get("metrics.test_f1"),
                "dataset_sha": run.get("tags.dataset_git_blob_sha", ""),
            }
        )
    return rows


def write_tracking_report() -> Path:
    """Document the tracked runs, the registry and the rollback procedure."""
    config.ensure_output_dirs()
    runs = list_runs()

    lines = [
        "# Experiment Tracking and Model Registry",
        "",
        "Closes gap **G5**. Produced by `src/tracking.py` using MLflow with the local file",
        "backend at `mlruns/mlflow.db`.",
        "",
        "## What this adds, precisely",
        "",
        "This project already had **reproducibility**: a fixed seed, pinned dependency versions,",
        "and a verified dataset SHA mean the same run produces the same artifact. MLflow does",
        "not add that, and saying it did would be wrong.",
        "",
        "What it adds is:",
        "",
        "1. **Comparability across runs** — every training run is recorded with its parameters,",
        "   metrics and dataset SHA, so two versions can be compared without reading a CSV by",
        "   hand.",
        "2. **Rollback** — the registry keeps every model version, so a regression can be undone",
        "   by re-pointing an alias rather than by retraining and hoping.",
        "3. **Attribution** — each run is tagged with the dataset blob SHA, so a metric can",
        "   always be traced to the exact data revision that produced it.",
        "",
        "## Scope limit — a deliberate decision",
        "",
        "MLflow runs against a **local SQLite backend** — one file, no server, no cloud. For a",
        "four-week project the demonstration value is identical to a hosted tracking server and",
        "the operational cost is not. This is recorded as a decision, not an omission.",
        "",
        "SQLite rather than the plain filesystem store because MLflow 3.x retired the file",
        "backend and raises unless `MLFLOW_ALLOW_FILE_STORE=true` is set. Opting out of a",
        "deprecation to keep a retired code path alive would be borrowing trouble; SQLite is the",
        "documented migration target and needs no infrastructure either.",
        "",
        "MLflow is a **development dependency**. It is never imported by the deployed",
        "application and is absent from the runtime image; a test asserts that separation, so",
        "the Space image stays small.",
        "",
        "## Tracked runs",
        "",
    ]

    if runs:
        lines += [
            "| Run | Version | Model | Test ROC-AUC | Test recall | Test F1 | Dataset SHA |",
            "|---|---|---|---:|---:|---:|---|",
        ]
        for run in runs:
            def fmt(value: Any) -> str:
                return f"{value:.4f}" if isinstance(value, (int, float)) else "—"

            lines.append(
                f"| `{run['run_id'][:8]}` | {run['version']} | {run['model']} | "
                f"{fmt(run['test_roc_auc'])} | {fmt(run['test_recall'])} | "
                f"{fmt(run['test_f1'])} | `{str(run['dataset_sha'])[:12]}` |"
            )
        lines += [
            "",
            f"**{len(runs)} run(s) tracked.** Each parent run contains nested child runs — one",
            "per candidate model — so the comparison that drove selection is recorded, not only",
            "the winner. Recording only the selected model would discard the evidence for why it",
            "was selected.",
            "",
        ]
    else:
        lines += [
            "No runs are recorded yet. Run `make track` after training.",
            "",
        ]

    lines += [
        "## Registry",
        "",
        f"Registered model: **`{REGISTERED_MODEL_NAME}`**",
        "",
        "The newest version carries the `production` alias. Aliases replaced stage transitions",
        "in MLflow 2.9+, and the alias is what a rollback re-points, so it is the meaningful",
        "handle rather than a label.",
        "",
        "## Rollback procedure",
        "",
        "If a newly promoted version regresses:",
        "",
        "```python",
        "import mlflow",
        "from mlflow.tracking import MlflowClient",
        "",
        f'mlflow.set_tracking_uri("{TRACKING_URI}")',
        "client = MlflowClient()",
        "",
        "# Inspect what exists",
        f'for v in client.search_model_versions("name=\'{REGISTERED_MODEL_NAME}\'"):',
        '    print(v.version, v.description)',
        "",
        "# Re-point production at the known-good version",
        f'client.set_registered_model_alias("{REGISTERED_MODEL_NAME}", "production", "1")',
        "",
        "# Retrieve it",
        f'model = mlflow.sklearn.load_model("models:/{REGISTERED_MODEL_NAME}@production")',
        "```",
        "",
        "Then export that pipeline over `deploy/artifacts/model_pipeline.joblib`, run",
        "`make test`, and push. The deployment workflow redeploys it automatically.",
        "",
        "**The rollback is deliberately not automated.** Re-pointing production at a different",
        "model is a decision with customer-facing consequences and should require a human, in",
        "keeping with the governance position taken everywhere else in this project.",
        "",
        "## Viewing the runs",
        "",
        "```bash",
        "make mlflow-ui",
        "```",
        "",
        "Then open <http://127.0.0.1:5000>.",
        "",
        "## Limitations",
        "",
        "- The local backend is single-user. Concurrent runs from several machines would need a",
        "  tracking server.",
        "- `mlruns/` is git-ignored, so run history is local to each machine and is not shared",
        "  through the repository. That is the correct trade-off here: experiment noise does not",
        "  belong in version control, and the artifacts that matter are committed under",
        "  `deploy/artifacts/`.",
        "- No automated promotion gate is implemented. Blocking promotion on a metric regression",
        "  is listed in Horizon 3.",
        "",
    ]

    path = config.REPORTS_DIR / "tracking_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MLflow tracking and model registry.")
    parser.add_argument("--log-current", action="store_true",
                        help="Retrain and log the run to MLflow.")
    parser.add_argument("--report", action="store_true", help="Write the tracking report.")
    args = parser.parse_args(argv)

    if args.log_current:
        from src.train import run_training  # noqa: PLC0415

        print("Training and logging to MLflow...")
        results = run_training()
        logged = log_training_run(results)
        print(f"\nParent run: {logged['run_id']}")
        for name, run_id in logged["candidates"].items():
            print(f"  candidate {name:<28} {run_id}")
        if "registered_version" in logged:
            print(f"Registered {REGISTERED_MODEL_NAME} version {logged['registered_version']} "
                  "with alias 'production'")

    path = write_tracking_report()
    runs = list_runs()
    print(f"\n{len(runs)} tracked run(s). Report -> {path}")
    print("View with: make mlflow-ui")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

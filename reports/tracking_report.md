# Experiment Tracking and Model Registry

Closes gap **G5**. Produced by `src/tracking.py` using MLflow with the local file
backend at `mlruns/mlflow.db`.

## What this adds, precisely

This project already had **reproducibility**: a fixed seed, pinned dependency versions,
and a verified dataset SHA mean the same run produces the same artifact. MLflow does
not add that, and saying it did would be wrong.

What it adds is:

1. **Comparability across runs** — every training run is recorded with its parameters,
   metrics and dataset SHA, so two versions can be compared without reading a CSV by
   hand.
2. **Rollback** — the registry keeps every model version, so a regression can be undone
   by re-pointing an alias rather than by retraining and hoping.
3. **Attribution** — each run is tagged with the dataset blob SHA, so a metric can
   always be traced to the exact data revision that produced it.

## Scope limit — a deliberate decision

MLflow runs against a **local SQLite backend** — one file, no server, no cloud. For a
four-week project the demonstration value is identical to a hosted tracking server and
the operational cost is not. This is recorded as a decision, not an omission.

SQLite rather than the plain filesystem store because MLflow 3.x retired the file
backend and raises unless `MLFLOW_ALLOW_FILE_STORE=true` is set. Opting out of a
deprecation to keep a retired code path alive would be borrowing trouble; SQLite is the
documented migration target and needs no infrastructure either.

MLflow is a **development dependency**. It is never imported by the deployed
application and is absent from the runtime image; a test asserts that separation, so
the Space image stays small.

## Tracked runs

| Run | Version | Model | Test ROC-AUC | Test recall | Test F1 | Dataset SHA |
|---|---|---|---:|---:|---:|---|
| `91051c6c` | 1.1.0 | Logistic Regression | 0.8414 | 0.7834 | 0.6130 | `3de7a612d160` |
| `633eff21` | 1.1.0 | Logistic Regression | 0.8414 | 0.7834 | 0.6130 | `3de7a612d160` |
| `f1274d25` | 1.1.0 | Logistic Regression | 0.8414 | 0.7834 | 0.6130 | `3de7a612d160` |

**3 run(s) tracked.** Each parent run contains nested child runs — one
per candidate model — so the comparison that drove selection is recorded, not only
the winner. Recording only the selected model would discard the evidence for why it
was selected.

## Registry

Registered model: **`churn-retention-classifier`**

The newest version carries the `production` alias. Aliases replaced stage transitions
in MLflow 2.9+, and the alias is what a rollback re-points, so it is the meaningful
handle rather than a label.

## Rollback procedure

If a newly promoted version regresses:

```python
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:////Users/krishnamathurm4pro/Desktop/Academics/SDIAM Term 3/SDAIM FINAL PROJECT/mlruns/mlflow.db")
client = MlflowClient()

# Inspect what exists
for v in client.search_model_versions("name='churn-retention-classifier'"):
    print(v.version, v.description)

# Re-point production at the known-good version
client.set_registered_model_alias("churn-retention-classifier", "production", "1")

# Retrieve it
model = mlflow.sklearn.load_model("models:/churn-retention-classifier@production")
```

Then export that pipeline over `deploy/artifacts/model_pipeline.joblib`, run
`make test`, and push. The deployment workflow redeploys it automatically.

**The rollback is deliberately not automated.** Re-pointing production at a different
model is a decision with customer-facing consequences and should require a human, in
keeping with the governance position taken everywhere else in this project.

## Viewing the runs

```bash
make mlflow-ui
```

Then open <http://127.0.0.1:5000>.

## Limitations

- The local backend is single-user. Concurrent runs from several machines would need a
  tracking server.
- `mlruns/` is git-ignored, so run history is local to each machine and is not shared
  through the repository. That is the correct trade-off here: experiment noise does not
  belong in version control, and the artifacts that matter are committed under
  `deploy/artifacts/`.
- No automated promotion gate is implemented. Blocking promotion on a metric regression
  is listed in Horizon 3.

# Licence and Attribution Notice

## Dataset

**IBM Telco Customer Churn sample** (`data/raw/Telco-Customer-Churn.csv`)

| Property | Value |
|---|---|
| Publisher | IBM |
| Source repository | https://github.com/IBM/telco-customer-churn-on-icp4d |
| Canonical raw URL | https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv |
| Repository licence | Apache License 2.0 |
| Licence text | https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/LICENSE |
| Verified Git blob SHA | `3de7a612d1609f25f21a455bda77948729369002` |

The file in `data/raw/` is byte-identical to the official published file and is never
modified by this project. All cleaning happens in code or in `data/processed/`.

> **The data represents a fictional telecommunications company.** It is not actual
> observed commercial customer data, and it must never be described or presented as
> such — including in the report, the slides, or the application.

IBM's documentation for the sample:
https://www.ibm.com/docs/en/cognos-analytics/12.0.x?topic=samples-telco-customer-churn

## Third-party software

This project depends on the following open-source packages. Each remains under its own
licence; none is vendored or redistributed here.

| Package | Licence |
|---|---|
| pandas | BSD 3-Clause |
| NumPy | BSD 3-Clause |
| scikit-learn | BSD 3-Clause |
| joblib | BSD 3-Clause |
| Matplotlib | Matplotlib licence (BSD-compatible, PSF-derived) |
| Streamlit | Apache License 2.0 |
| pytest | MIT |
| nbformat, nbclient, ipykernel | BSD 3-Clause |

Base container image: `python:3.11-slim` (Docker Official Image; Python is under the PSF
licence, Debian components under their respective licences).

GitHub Actions used:

| Action | Publisher |
|---|---|
| `actions/checkout` | GitHub (MIT) |
| `actions/setup-python` | GitHub (MIT) |
| `huggingface/hub-sync` | Hugging Face |

## Academic work

The source code, documentation, analysis and application in this repository were produced
for an academic assignment. Reuse of the dataset remains subject to IBM's Apache-2.0 terms
above.

## Model artifact

`deploy/artifacts/model_pipeline.joblib` is derived from the IBM sample. It inherits the
dataset's limitations: it reflects a fictional population and carries no warranty of
performance on any real customer base. See `deploy/artifacts/model_card.md`.

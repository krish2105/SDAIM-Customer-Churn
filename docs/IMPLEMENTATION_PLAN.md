# Implementation Plan

Phases, acceptance criteria and the exact command sequence. Phases 0–8 are complete
locally; phases 9–11 require the user's platform credentials and cannot be performed here.

---

## Phase 0 — Audit and integrity (COMPLETE)

**Goal.** Establish what exists and prove the dataset is the official file before any code
depends on it.

| Acceptance criterion | Result |
|---|---|
| Every attached input read | Met — brief, data dictionary, manifest, CSV, template |
| `docs/INPUT_AUDIT.md` written | Met |
| Git blob SHA equals `3de7a61…9002` | Met — exact match |
| 7,043 rows × 21 columns, exact names and order | Met |
| `customerID` unique, `Churn` ∈ {Yes, No}, `SeniorCitizen` ⊆ {0,1} | Met |
| Missing inputs recorded, not invented | Met — `SOURCE_MANIFEST.json` located in the sibling package and copied; unknown values marked `<<UNRESOLVED>>` |

```bash
git hash-object Telco-Customer-Churn.csv
python verify_ibm_telco_dataset.py Telco-Customer-Churn.csv
```

---

## Phase 1 — Environment and skeleton (COMPLETE)

| Acceptance criterion | Result |
|---|---|
| Python 3.11 virtual environment | Met — 3.11.15 |
| Repository structure created | Met |
| Dependencies install cleanly | Met |
| `.gitignore` covers credentials and caches | Met |

```bash
bash scripts/bootstrap_macos.sh
```

---

## Phase 2 — Validation module (COMPLETE)

| Acceptance criterion | Result |
|---|---|
| 12 documented check families implemented | Met — 30 individual checks |
| JSON report written to `reports/tables/` | Met |
| Raw file never mutated | Met — read-only access throughout |
| Fails loudly with a precise report | Met — non-zero exit and per-check detail |

```bash
make validate
```

---

## Phase 3 — Exploratory analysis (COMPLETE)

| Acceptance criterion | Result |
|---|---|
| All 12 required analyses | Met |
| ≥ 11 figures at 200 dpi, deterministic filenames | Met — 11 figures |
| Summary tables exported | Met — 13 CSV tables |
| Every quantitative claim computed, none hand-entered | Met — `write_observations` consumes computed frames only |

```bash
make eda
```

---

## Phase 4 — Preprocessing and modelling (COMPLETE)

| Acceptance criterion | Result |
|---|---|
| Single split before any transformer is fitted | Met |
| All transformers inside the `Pipeline` | Met |
| Two substantive models plus a baseline | Met |
| Selection rule fixed in code before test evaluation | Met — see D-08 |
| Test set used exactly once | Met |
| All required metrics and figures produced | Met |

```bash
make train
```

Measured outcome: Logistic Regression selected; test ROC-AUC 0.8414, recall 0.7834,
F1 0.6130, precision 0.5034, accuracy 0.7374.

---

## Phase 5 — Artifacts (COMPLETE)

| Acceptance criterion | Result |
|---|---|
| Complete pipeline serialised | Met — 8.4 KB, so no Git LFS needed |
| Metadata holds real numbers, no placeholders | Met — enforced by test |
| Feature schema with categories, bounds, labels, grouping | Met |
| Model card documenting use, limits and governance | Met |
| Artifact loads in a fresh process | Met |

---

## Phase 6 — Application (COMPLETE)

| Acceptance criterion | Result |
|---|---|
| Loads artifacts via `pathlib`, cached, never trains | Met |
| Controls for all 19 predictors in 4 groups | Met |
| Dependency rules enforced | Met — phone and internet locks verified in browser |
| Non-negative validation | Met |
| Class, probability, risk tier, cautious action | Met |
| Model information and limitations expander | Met |
| No stack traces or paths shown to the user | Met |
| Light/dark toggle, professional design, charts | Met — verified in both themes |

```bash
make app
```

---

## Phase 7 — Tests and packaging (COMPLETE)

| Acceptance criterion | Result |
|---|---|
| All 17 required test areas covered | Met — 52 tests |
| Full suite passes | Met |
| Dockerfile per specification | Met |
| Image builds, runs and answers the health check | Met |
| Container reproduces the local prediction exactly | Met — 22.9% in both |

```bash
make test
make docker-run
```

---

## Phase 8 — CI/CD, security and documentation (COMPLETE)

| Acceptance criterion | Result |
|---|---|
| `ci.yml` with all eight required steps | Met |
| `deploy.yml` with validation, least privilege, concurrency, config checks | Met |
| Action inputs verified against published definitions | Met — found a required input the spec omitted (D-19) |
| Secret scan passes | Met |
| All documentation written | Met |
| Local commits after tests pass | Met |

```bash
make secret-scan
make verify
```

---

## Phase 9 — GitHub (PENDING — requires the user)

| Acceptance criterion | How to verify |
|---|---|
| Repository exists and is accessible | Open the URL |
| `main` contains the full project | Compare the file tree |
| CI run succeeds | Open the run, confirm every step green |
| `HF_TOKEN` secret exists | Settings page shows the **name** only |
| `HF_SPACE_ID` variable set correctly | Settings page shows name and value |

```bash
git remote add origin https://github.com/krish2105/MLops-Huggingface.git
git branch -M main
git push -u origin main
```

---

## Phase 10 — Hugging Face Space (PENDING — requires the user)

| Acceptance criterion | How to verify |
|---|---|
| Docker Space created, ID recorded | Space settings page |
| Deployment workflow succeeds | Actions run page |
| Space build succeeds | Space build logs |
| Live application loads and predicts | Use the live app |

---

## Phase 11 — Visible-change deployment test (PENDING — requires the user)

| Acceptance criterion | How to verify |
|---|---|
| Version raised 1.0.0 → 1.1.0 | `MODEL_VERSION` in `src/config.py`, retrain, confirm metadata |
| Visible caption added | Appears in the running app |
| Tests and secret scan pass before committing | Local output |
| Second workflow run succeeds | Actions run page |
| Space rebuilds and shows 1.1.0 | Live application |
| Predictions still work | Live application |

Full procedure in `docs/DEMONSTRATION_SCRIPT.md` §"Visible-change deployment test".

---

## Command sequence, start to finish

```bash
# Local (all verified)
bash scripts/bootstrap_macos.sh
make validate
make eda
make train
make notebook
make test
make secret-scan
make app            # http://localhost:8501
make docker-run     # http://localhost:7860
make verify

# External (requires your credentials — run these yourself)
git remote add origin <your repository URL>
git branch -M main
git push -u origin main
```

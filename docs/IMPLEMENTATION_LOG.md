# Implementation Log

Chronological record of commands run, results, decisions and unresolved external steps.
No credential is recorded here, and none was created during this build.

All work performed on macOS (Darwin 25.5.0), Apple silicon, 2026-07-23.

---

## 1. Audit and integrity

| Command | Result |
|---|---|
| `ls -la` in the project folder | 6 input files found. `SOURCE_MANIFEST.json` and `verify_ibm_telco_dataset.py` were **not** present — located in the sibling `IBM_Telco_Claude_Code_Package/` and the manifest copied in unchanged |
| Extracted `Final Group Project.docx` | **PASS** — full brief recovered, all five parts and their tasks |
| Read `IBM_Telco_Customer_Churn_Data_Dictionary.csv` | **PASS** — column roles, expected values and UI guidance recovered |
| `git hash-object Telco-Customer-Churn.csv` | **PASS** — `3de7a612d1609f25f21a455bda77948729369002`, exact match |
| `shasum -a 256` | `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` (recorded, informational) |
| `python3.11 verify_ibm_telco_dataset.py` | **PASS** — "Dataset validation passed", exit 0 |
| Stdlib profiling of the CSV | **PASS** — 7,043 rows; Churn No 5,174 / Yes 1,869; 11 blank `TotalCharges`, all `tenure = 0`; 0 duplicate rows; all 16 category domains match the data dictionary |

**Decision.** Proceed. The dataset is the exact official revision.

**Environment observed.** Python 3.11.15 at `/opt/homebrew/bin/python3.11`; default `python3`
is Anaconda 3.13.5 and was **not** used. Git 2.50.1. Docker CLI 29.6.1 present but the
daemon was **not running** at audit time. Network reachable.

**Open question raised with the user.** The brief says "real-world dataset"; the IBM sample
is fictional. Recorded as decision D-01 and as pending instructor confirmation.

---

## 2. Questions put to the user

Four decisions were genuinely the user's, so they were asked before implementation:

| Question | Answer |
|---|---|
| Front-end ambition | Custom design system with light/dark toggle **and** in-app charts (options 1 + 3) |
| Docker gate | User to start Docker; a real build and run to be performed |
| Git | Initialise and create staged local commits; **no push** |
| Academic details | User will supply; marked `<<UNRESOLVED>>` meanwhile |

The academic details were not supplied during the build, so every affected file carries an
explicit `<<UNRESOLVED>>` marker. **Nothing was invented.**

---

## 3. Environment

| Command | Result |
|---|---|
| `python3.11 -m venv .venv` | **PASS** |
| `pip install -r requirements-dev.txt` | **PASS** |
| Version check | Python 3.11.15, pandas 3.0.5, numpy 2.4.6, scikit-learn 1.9.0, joblib 1.5.3, matplotlib 3.11.1, streamlit 1.60.0, pytest 9.1.1 |

---

## 4. Validation and EDA

| Command | Result |
|---|---|
| `python -m src.data_validation` | **PASS** — 30 checks, report written to `reports/tables/data_validation_report.json` |
| `python -m src.eda` | **PASS** — 11 figures, 13 tables, `reports/eda_observations.md` |

Measured: churn 26.54%; contract month-to-month 42.71% / one year 11.27% / two year 2.83%;
electronic check 45.29%; fibre optic 41.89%; no tech support 41.64%; churner mean tenure
17.98 months against 37.57; tenure–churn correlation −0.352; tenure–TotalCharges 0.826.

---

## 5. Training

| Command | Result |
|---|---|
| `python -m src.train` | **PASS** |

- Split: 5,634 train / 1,409 test; churn rate 0.2654 in both.
- CV (training split only): Logistic Regression ROC-AUC 0.8460 ± 0.0124, F1 0.6286, recall
  0.8013. Random Forest 0.8454 ± 0.0091, F1 0.6296, recall 0.7632. Dummy 0.5065.
- **Selection:** ROC-AUC gap 0.0005 and F1 gap 0.0010 both inside the 0.01 tolerance, so the
  pre-declared rule fell through to CV recall → **Logistic Regression**.
- Held-out: accuracy 0.7374, precision 0.5034, recall 0.7834, F1 0.6130, ROC-AUC 0.8414.
  Confusion matrix TN 746 / FP 289 / FN 81 / TP 293.
- Random Forest held-out: accuracy 0.7608, precision 0.5337, recall 0.7834, F1 0.6349,
  ROC-AUC 0.8417.

**Recorded honestly:** the Random Forest scored marginally better on three held-out metrics.
The model was **not** switched, because reselecting after seeing test results would
invalidate the held-out evaluation. See D-08.

**Artifact size:** `model_pipeline.joblib` = 8.4 KB → **Git LFS not required.** Checked, not
assumed.

**Re-run after adding `reference_rates.json`:** identical numbers, confirming determinism at
`random_state=42`.

---

## 6. Application development

| Step | Result |
|---|---|
| Built `theme.py` (design tokens, CSS), `charts.py`, `app.py` | Done |
| First browser check, light mode | **FAIL** — two real defects found |
| Fix 1 | Streamlit followed the OS colour scheme and fought the toggle → base theme pinned in `deploy/.streamlit/config.toml` (D-16) |
| Fix 2 | Section cards rendered empty with widgets below them — Streamlit closes unbalanced HTML per markdown block → switched to `st.container(border=True)` (D-17) |
| Second browser check, dark mode | **FAIL** — three further defects |
| Fix 3 | A broad `font-family` rule on `span`/`div` overrode Streamlit's **icon font**, so ligature names rendered as literal text (`_arrow_right`) → scoped to `.stApp` |
| Fix 4 | Select and number-input surfaces stayed light in dark mode; this Streamlit build exposes no BaseWeb attributes → re-targeted via `:has(> [role="combobox"])` and `[data-testid="stNumberInputContainer"]`, confirmed against the live DOM |
| Fix 5 | The result vanished when the theme was toggled → assessment persisted in session state with a staleness notice (D-18) |
| Nested expander glitch | Removed; "Full model card" made a sibling expander |
| Charts cramped in the narrow column | Moved to a full-width row beneath both columns |
| Chart polish | Gauge marker confined to its band; segment-chart legend moved off the last bar; `set_axisbelow(True)` so grid lines sit behind the data |
| Final browser check | **PASS** — both themes, prediction 22.9% Low risk, three context charts, dependency locks working |

---

## 7. Tests

| Command | Result |
|---|---|
| `pytest` (first run) | **1 failed, 51 passed** |
| Failure | `test_metadata_reports_a_real_selected_model` — the assertion `"ACTUAL" not in metadata` matched the legitimate phrase "not **actual** observed commercial customer data" |
| Fix | Assert against specific placeholder tokens (`ACTUAL_NUMBER`, `TODO`, `UNRESOLVED`, `<<`) instead of a bare substring |
| `pytest` (after fix) | **PASS — 52 passed** |

---

## 8. Docker

| Command | Result |
|---|---|
| `docker info` | Daemon **running** (user started it as agreed) |
| `bash scripts/build_docker.sh --run` | **PASS** — image built, container started, health check passed on the 2nd attempt |
| Containerised prediction check | **PASS** — the container returned **22.9%**, identical to the local environment, confirming the pinned dependencies and artifact are consistent |

---

## 9. Secret scanning

| Command | Result |
|---|---|
| `bash scripts/scan_secrets.sh` (before docs) | **PASS** |
| `pytest` after writing `docs/SECURITY.md` | **FAIL** — the scanner flagged its own documentation: `aws_secret_key`, `aws_session_token` matched the pattern table in SECURITY.md |
| Fix | Both the shell scanner and the test now require a credential-shaped value (16+ base64 characters) rather than any non-space token. A scanner that cries wolf on its own docs gets muted, which is worse than a slightly narrower pattern |
| **Canary verification** | A temporary file containing four synthetic credentials (AWS key id, AWS secret, AWS session token, HF token) was placed in the project. **Both** the shell scanner and pytest **correctly failed**, naming only the file and category — never the matched text. After removal both passed again |

---

## 10. Workflow verification

| Check | Result |
|---|---|
| Fetched `huggingface/hub-sync` action definition | **Found a required input the specification omitted:** `github_repo_id`. Added as `${{ github.repository }}` — without it the deploy job would have failed at run time |
| Confirmed `hf_token` is the correct token input name | **PASS** |
| Checked released versions | `hub-sync` latest v0.2.1 (2026-07-15) → used. `actions/checkout` kept at v6 per specification; v7 released three days before this build and cannot be validated here. `actions/setup-python` v6 (latest is v7, three days old) |

---

## 11. Git

| Command | Result |
|---|---|
| `git init -b main` | **PASS** |
| Pre-commit gate before each commit | `pytest` + `scan_secrets.sh` + `git diff --cached --name-only` reviewed |
| Commits created | 6 staged commits (see `git log`) |
| Tracked sensitive filenames | **None** — verified by `git ls-files` pattern check |
| **Push** | **NOT performed.** No remote configured; awaiting explicit approval |

Build inputs excluded from version control with a documented rationale: the duplicate
root-level CSV (canonical copy is `data/raw/`), the instructor brief and the build
specification (third-party documents). The data dictionary is versioned as
`docs/DATA_DICTIONARY.csv`.

---

## 12. Release verification

| Run | Result |
|---|---|
| First `scripts/verify_release.sh` | **8 PASS, 2 FAIL** |
| Failure 1 | `docs/IMPLEMENTATION_LOG.md` genuinely missing — this file. Written |
| Failure 2 | Streamlit smoke test. Two real causes: (a) `.streamlit/config.toml` set `enableCORS = false` alongside `enableXsrfProtection = true`, which Streamlit rejects and overrides with a warning — the CORS line was removed and XSRF protection kept; (b) the gate killed the launching subshell, not the Streamlit process, so an orphan held the port on the next run. The gate now reclaims the port first, detects early process death, and cleans up by port pattern |
| Final run | Recorded in `docs/QUALITY_GATE_RESULTS.md` |

---

## 13. Unresolved external steps

None of these can be completed here; each requires the user's credentials and direct
observation.

1. Create the GitHub repository and push `main`.
2. Create the Hugging Face Docker Space and record its exact ID.
3. Create a fine-grained HF write token scoped to that Space.
4. Add the `HF_TOKEN` secret and the `HF_SPACE_ID` variable in GitHub.
5. Observe a successful CI run and record its real URL, ID and timestamp.
6. Observe a successful deployment run and Space build.
7. Verify the live application predicts.
8. Perform the visible-change test (1.0.0 → 1.1.0) and verify the second run and the live
   version.
9. Capture screenshots 23–37.
10. Supply the academic details still marked `<<UNRESOLVED>>`.

**No deployment has occurred. No external platform evidence exists yet, and none is
claimed.**

---

## 14. Deviations from the master specification

| Deviation | Reason |
|---|---|
| Added `deploy/theme.py` and `deploy/charts.py` | The requested design system and in-app charts; keeping them in `app.py` would have made it unreadable |
| Added `deploy/artifacts/reference_rates.json` | Descriptive statistics for the context charts. Participates in no prediction (D-15) |
| Added `deploy/.streamlit/config.toml` | Required to make the light/dark toggle behave deterministically (D-16) |
| Added `scripts/build_notebook.py` | Generates and **executes** the notebook, so committed cells have genuinely run |
| Added `docs/DATA_DICTIONARY.csv` | Versioned provenance documentation (deliverable 2) |
| `hub-sync` v0.2.1 instead of v0.1.0, plus `github_repo_id` | Latest release; the extra input is **required** by the action (D-19) |
| Root-level build inputs git-ignored | Duplicate dataset and third-party documents |

Every deviation is recorded in `docs/DECISIONS.md`.

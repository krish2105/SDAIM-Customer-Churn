# Input Audit

Project: Customer Churn Intelligence and Retention Decision-Support Platform
Audit performed before any application code was written.

Working directory: `/Users/krishnamathurm4pro/Desktop/Academics/SDIAM Term 3/SDAIM FINAL PROJECT`

---

## 1. Files found in the project folder

| File | Status | Notes |
|---|---|---|
| `Final Group Project.docx` | Found, readable | Instructor brief. Text extracted successfully from the OOXML package. |
| `CLAUDE_CODE_MASTER_PROMPT.md` | Found | Build specification supplied by the user. |
| `Telco-Customer-Churn.csv` | Found, validated | Official IBM sample. Copied unmodified to `data/raw/`. |
| `IBM_Telco_Customer_Churn_Data_Dictionary.csv` | Found, readable | Column roles, expected values, preprocessing and UI guidance. |
| `IBM_Telco_Customer_Churn_Data_Dictionary.xlsx` | Found | Same content as the CSV export; the CSV was used as the machine-readable source. |
| `PROJECT_INPUTS_TEMPLATE.md` | Found | Template only — most academic and deployment values are still blank. |

## 2. Files referenced by the brief but NOT present in the project folder at audit time

| Expected file | Status | Resolution |
|---|---|---|
| `SOURCE_MANIFEST.json` | Not in project folder | Located in the sibling build package `../IBM_Telco_Claude_Code_Package/`. Copied into the project root unchanged. |
| `PROJECT_INPUTS.md` | Not present | Only `PROJECT_INPUTS_TEMPLATE.md` exists. A `PROJECT_INPUTS.md` is created from the template with **unknown values left explicitly marked as unresolved**. No value was invented. |
| `verify_ibm_telco_dataset.py` | Not in project folder | Located in the sibling build package. Executed from that location for the first-pass check; an equivalent, extended validator is implemented in `src/data_validation.py`. |

## 3. Non-secret project inputs recovered

From `SOURCE_MANIFEST.json`:

- Dataset name: `Telco-Customer-Churn.csv`
- Publisher: IBM
- Business context: **fictional** telecommunications company customer churn sample
- Official repository: `https://github.com/IBM/telco-customer-churn-on-icp4d`
- Official raw URL: `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
- Repository licence: Apache-2.0
- Expected data rows: 7043; expected columns: 21
- Expected Git blob SHA: `3de7a612d1609f25f21a455bda77948729369002`
- Target: `Churn`; identifier excluded from modelling: `customerID`

From `PROJECT_INPUTS_TEMPLATE.md`:

- GitHub repository (stated, **not yet verified to exist**): `https://github.com/krish2105/MLops-Huggingface`
- GitHub default branch: `main`
- Desired Space name: `customer-churn-intelligence`

No credential, token, or key was found in any input file, and none was written to this repository.

## 4. Unresolved placeholders

These remain unknown. They are **not** invented anywhere in this project; every document that needs them carries an explicit `UNRESOLVED` marker.

- Course/module code, group number, group member names, student IDs, instructor name, submission date
- Hugging Face username or organisation
- Final Hugging Face Space ID (`username-or-org/space-name`)
- Space visibility (public / protected / private)
- Instructor confirmation that the IBM fictional sample satisfies the brief's "real-world dataset" wording
- Required final report format, presentation duration, required repository visibility
- Existence and accessibility of the stated GitHub repository
- All external deployment evidence: commit SHAs, Actions run IDs, Space URL, screenshots, timestamps

## 5. Instructor brief readability

**Readable.** `Final Group Project.docx` was parsed and the full task list recovered. Key requirements extracted:

- Part 1: dataset selection and EDA; preprocessing with leakage prevention; train and compare **at least two** models; save the model (Joblib) together with preprocessing components or the complete pipeline.
- Part 2: build an interactive Streamlit app with input fields, a predict action, clear output and probability/confidence; test locally.
- Part 3: deploy to a Hugging Face Space with all runtime files.
- Part 4: GitHub repository; HF token stored as a GitHub Actions **secret**, never hard-coded; workflow triggered on push to `main` that syncs files to the Space; then a **visible-change** test proving automated redeployment.
- Part 5: report with problem statement, dataset source, EDA, preprocessing, model comparison, selection justification, app screenshots, repo structure, workflow explanation, workflow-run screenshot, Space screenshot, and both links.

Point of tension recorded for instructor confirmation: the brief asks for a **"real-world dataset"**. The IBM Telco sample is a widely used, officially published dataset representing a **fictional** telecommunications company. It is realistic in structure but is not observed commercial customer data. This project describes it accurately as a fictional sample throughout and requests instructor confirmation. See `docs/DECISIONS.md`.

## 6. Dataset integrity validation

Executed with Python 3.11.15 against `Telco-Customer-Churn.csv`.

| Check | Required | Observed | Result |
|---|---|---|---|
| Git blob SHA | `3de7a612d1609f25f21a455bda77948729369002` | `3de7a612d1609f25f21a455bda77948729369002` | PASS |
| SHA-256 (recorded, informational) | `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` | identical | PASS |
| Data rows | 7043 | 7043 | PASS |
| Columns | 21, exact names and order | 21, exact match | PASS |
| `customerID` unique | yes | yes | PASS |
| `Churn` domain | exactly {Yes, No} | {Yes, No} | PASS |
| `SeniorCitizen` domain | subset of {0, 1} | {0, 1} | PASS |
| Duplicate full rows | 0 expected | 0 | PASS |
| Blank `TotalCharges` | tolerated, must be handled in code | 11 rows, all with `tenure = 0` | PASS (handled by coercion + median imputation fitted on training data only) |
| Negative values in `tenure` / `MonthlyCharges` / `TotalCharges` | none | none | PASS |
| Category domains vs data dictionary | exact | exact for all 16 categorical predictors | PASS |

Observed ranges: `tenure` 0–72; `MonthlyCharges` 18.25–118.75; `TotalCharges` 18.80–8684.80.
Observed target balance: `No` 5174, `Yes` 1869 (26.54% positive class).

The supplied verifier `verify_ibm_telco_dataset.py` also reported `Dataset validation passed` (exit code 0).

**Conclusion: the dataset passed integrity validation and implementation may proceed.** The raw file in `data/raw/` is byte-identical to the audited input and is never mutated.

## 7. Environment observed

| Component | Observed |
|---|---|
| Platform | macOS (Darwin 25.5.0), Apple silicon |
| Python 3.11 | 3.11.15 at `/opt/homebrew/bin/python3.11` — used as the project runtime |
| Default `python3` | 3.13.5 (Anaconda) — **not** used for this project |
| Git | 2.50.1; the project folder was **not** a Git repository at audit time |
| Docker | CLI 29.6.1 present; **daemon not running** at audit time |
| Network | Reachable |

Consequence recorded: Docker build/run gates can only be executed once the Docker daemon is started. Until then the Dockerfile is validated statically and the gate is reported as `NOT RUN`, never as passed.

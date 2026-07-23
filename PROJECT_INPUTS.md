# Project Inputs

Derived from `PROJECT_INPUTS_TEMPLATE.md`. Values that are genuinely unknown are marked
`<<UNRESOLVED>>` and **have not been invented**. Fill them in before submission.

No token, key or password may ever be written in this file.

## Academic details

| Field | Value |
|---|---|
| Course / module | `<<UNRESOLVED>>` |
| Group number | `<<UNRESOLVED>>` |
| Group member names | `<<UNRESOLVED>>` |
| Student IDs | `<<UNRESOLVED>>` |
| Instructor | `<<UNRESOLVED>>` |
| Submission date | `<<UNRESOLVED>>` |

## Repository and deployment

| Field | Value | Verified? |
|---|---|---|
| GitHub repository | `https://github.com/krish2105/MLops-Huggingface` | **Not verified** — stated in the input template; existence and access were never confirmed by this build |
| GitHub default branch | `main` | Not verified |
| Hugging Face username or organisation | `<<UNRESOLVED>>` | — |
| Desired Space name | `customer-churn-intelligence` | Preference only |
| Intended Space ID (`username-or-org/space-name`) | `<<UNRESOLVED>>` | — |
| Space visibility | `<<UNRESOLVED: public / protected / private>>` | — |

## Dataset — all verified

| Field | Value | Verified |
|---|---|---|
| Attached CSV filename | `Telco-Customer-Churn.csv` | Yes |
| Official source URL | `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv` | Recorded from `SOURCE_MANIFEST.json` |
| Data rows | 7,043 | Yes — counted |
| Columns | 21 | Yes — counted, exact names and order |
| Git blob SHA | `3de7a612d1609f25f21a455bda77948729369002` | Yes — `git hash-object` matched exactly |
| SHA-256 (informational) | `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` | Yes |
| Licence of source repository | Apache-2.0 | Recorded from `SOURCE_MANIFEST.json` |

## Decisions requiring instructor confirmation

| Question | Status |
|---|---|
| Does the IBM **fictional** telco sample satisfy the brief's "real-world dataset" wording? | **Pending.** The dataset is officially published by IBM and widely used, but represents a fictional company. This project describes it accurately throughout. See `docs/DECISIONS.md`, decision D-01. |
| Required final report format (PDF / DOCX / both) | `<<UNRESOLVED>>` |
| Required presentation duration | `<<UNRESOLVED>>` — `docs/DEMONSTRATION_SCRIPT.md` is written for five minutes |
| Required repository visibility | `<<UNRESOLVED>>` |

## Secret handling

No token appears anywhere in this repository. Create these only in the platform settings:

| Type | Name | Where |
|---|---|---|
| GitHub Actions **secret** | `HF_TOKEN` | Settings → Secrets and variables → Actions → Secrets |
| GitHub Actions **repository variable** | `HF_SPACE_ID` | Settings → Secrets and variables → Actions → Variables |

See `docs/SECURITY.md` for the full procedure, including token scoping and rotation.

## Build environment actually used

| Component | Value |
|---|---|
| Platform | macOS (Darwin 25.5.0), Apple silicon |
| Python | 3.11.15 |
| scikit-learn / pandas / numpy | 1.9.0 / 3.0.5 / 2.4.6 |
| Streamlit | 1.60.0 |
| Docker | 29.6.1 — image built and health-checked locally |
| Git | 2.50.1 |

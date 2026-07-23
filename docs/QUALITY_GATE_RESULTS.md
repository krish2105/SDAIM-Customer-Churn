# Quality Gate Results

Actual recorded output from `bash scripts/verify_release.sh`.

| Field | Value |
|---|---|
| Run started (UTC) | 2026-07-23T15:00:27Z |
| Run finished (UTC) | 2026-07-23T15:00:38Z |
| Host | macOS (Darwin 25.5.0), Apple silicon |
| Python | 3.11.15 (`.venv`) |
| Result | **10 PASS · 0 FAIL · 0 NOT RUN** — script exit code 0 |

> Total wall time is short because the Docker layer cache was warm from an earlier build in
> the same session. Every gate executed; the per-gate evidence below is quoted from the run.

---

## Gate results

### 1. Dataset validation — PASS

```
COMMAND: .venv/bin/python -m src.data_validation --strict-sha
File            : data/raw/Telco-Customer-Churn.csv
Rows / columns  : 7043 / 21
Git blob SHA    : 3de7a612d1609f25f21a455bda77948729369002
Expected SHA    : 3de7a612d1609f25f21a455bda77948729369002
Target counts   : {'No': 5174, 'Yes': 1869}
Blank strings   : {'TotalCharges': 11}
Checks executed : 30
RESULT          : PASSED
```

Run with `--strict-sha`, so an unavailable Git would have failed the gate rather than
skipping it.

### 2. Python syntax compilation — PASS

```
COMMAND: .venv/bin/python -m compileall -q src tests deploy
```

All modules under `src`, `tests` and `deploy` compile.

### 3. Artifact reload in a fresh process — PASS

```
Loaded Logistic Regression v1.0.0
Scored the schema default record: probability=0.210696
```

A separate interpreter loaded `model_pipeline.joblib`, built a row from
`feature_schema.json`, produced a probability inside [0, 1], and confirmed every metric in
the metadata is numeric rather than a placeholder string.

### 4. Test suite — PASS

```
COMMAND: .venv/bin/python -m pytest -q
52 passed in 2.07s
```

| Module | Tests |
|---|---:|
| `test_data_validation.py` | 13 |
| `test_model_artifact.py` | 12 |
| `test_prediction.py` | 11 |
| `test_deployment_files.py` | 16 |
| **Total** | **52** |

### 5. Streamlit smoke test — PASS

```
Streamlit answered /_stcore/health with 200 on port 8599
```

The real server was started against the exported artifacts and answered its health
endpoint. Not a mock.

### 6. Secret scan — PASS

```
COMMAND: bash scripts/scan_secrets.sh
SECRET SCAN PASSED: no credential patterns detected.
```

**The scanner was itself verified.** A temporary file containing four synthetic credentials
(AWS access key id, AWS secret access key, AWS session token, Hugging Face token) was placed
in the project. Both the shell scanner and the pytest equivalent correctly **failed**,
reporting only the file name and pattern category — never the matched text. Both passed
again once the file was removed. A scanner that has never been shown to fail is not
evidence of anything.

### 7. Required-file verification — PASS

```
Figures present: 16
All required files present.
```

47 required paths checked, covering documentation, source, tests, reports, the deployment
package, artifacts and workflows.

### 8. Docker build — PASS

```
COMMAND: docker build -t churn-intelligence:verify deploy/
```

Built from `deploy/Dockerfile` (`python:3.11-slim`, non-root `appuser`, port 7860).

### 9. Docker run and health check — PASS

```
Container answered /_stcore/health with 200 on port 7861
```

The container was started, polled until healthy, and removed.

**Additional check performed separately:** the containerised application returned a churn
probability of **22.9%** for the default input — **identical** to the local environment,
confirming the artifact and pinned dependencies behave the same inside the image.

### 10. Git status and staged-file review — PASS

```
--- tracked files matching sensitive names (expect none) ---
none
```

No tracked file matches `.env`, `credentials`, `*.pem`, `id_rsa` or `id_ed25519`.

---

## Earlier run — recorded for transparency

The first execution of this script produced **8 PASS, 2 FAIL**. Both failures were real and
both were fixed rather than suppressed.

| Failure | Cause | Fix |
|---|---|---|
| Required-file verification | `docs/IMPLEMENTATION_LOG.md` did not exist yet | Written |
| Streamlit smoke test | Two causes. (a) `deploy/.streamlit/config.toml` set `enableCORS = false` alongside `enableXsrfProtection = true`; Streamlit rejects that combination and overrides it with a warning. (b) The gate killed the launching subshell rather than the Streamlit process, so an orphan held the port on the next run and the gate could answer healthy from the wrong process | The CORS line was removed and XSRF protection kept. The gate now reclaims the port before starting, aborts early if the server dies, and cleans up by port pattern |

Two further test failures were found and fixed during development, both recorded in
`docs/IMPLEMENTATION_LOG.md`:

- `test_metadata_reports_a_real_selected_model` matched the word "actual" inside the
  legitimate phrase *"not actual observed commercial customer data"*. The assertion now
  targets specific placeholder tokens.
- The secret-pattern test flagged `docs/SECURITY.md`, which documents the very patterns it
  searches for. Both scanners now require a credential-shaped value.

---

## External gates — NOT RUN, and NOT passed

These cannot be executed here. They require the user's platform credentials and must be
verified by direct observation. **No deployment has occurred, and none is claimed.**

| Gate | Status | Evidence required |
|---|---|---|
| GitHub repository push | ⬜ Not run | Repository URL; real commit SHAs |
| GitHub Actions CI run | ⬜ Not run | Run URL, run ID, timestamp, all steps green |
| GitHub Actions deployment run | ⬜ Not run | Run URL and timestamp |
| Hugging Face Space build | ⬜ Not run | Build log showing success |
| Live application prediction | ⬜ Not run | Screenshot of a prediction on the Space |
| Visible-change redeployment (1.0.0 → 1.1.0) | ⬜ Not run | Second run plus version 1.1.0 visible live |

Local gates passing says the code is correct and the container runs. It says nothing about
whether a deployment succeeded. Those are separate claims requiring separate evidence.

---

## Reproducing this run

```bash
make verify
```

Re-run and refresh this document after any material change, and again before submission.

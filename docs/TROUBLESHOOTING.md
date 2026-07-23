# Troubleshooting

Each entry: the symptom, why it happens, and the fix. Several of these were encountered
and resolved during this build; those are marked **(observed)**.

---

## 1. Missing dataset

**Symptom**
```
VALIDATION FAILED: Raw dataset not found at .../data/raw/Telco-Customer-Churn.csv
```

**Cause.** The raw file is absent, renamed, or the command was run from the wrong
directory.

**Fix.** Restore the official file — do not substitute another dataset:

```bash
curl -L -o data/raw/Telco-Customer-Churn.csv \
  https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
git hash-object data/raw/Telco-Customer-Churn.csv
# must print 3de7a612d1609f25f21a455bda77948729369002
make validate
```

If the SHA differs, you have a different revision. Do not proceed: every metric in the
report is tied to this exact file.

---

## 2. Schema mismatch

**Symptom**
```
VALIDATION FAILED ... columns_exact_and_ordered
```
or a category check naming an unexpected level.

**Cause.** The CSV was opened and re-saved (Excel is the usual culprit — it reorders
columns, changes encodings, and reformats numbers), or a different file was downloaded.

**Fix.** Never edit the raw file. Re-download as above. If you genuinely need a modified
dataset, write the modified copy to `data/processed/` and leave `data/raw/` untouched.

Inspect what changed:

```bash
python -c "import pandas as pd; print(list(pd.read_csv('data/raw/Telco-Customer-Churn.csv').columns))"
```

---

## 3. `TotalCharges` conversion

**Symptom.** `ValueError: could not convert string to float: ' '`, or a spuriously high
missing-value count.

**Cause.** `TotalCharges` is stored as text and 11 rows hold a blank — customers with
`tenure = 0` who have not completed a billing cycle. Reading it naively as numeric fails or
produces surprises.

**Fix.** Use the documented rule, which the project applies everywhere:

```python
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
```

Do **not** fill the blanks before splitting. The median imputer lives inside the pipeline
and is fitted on the training split only; imputing beforehand is data leakage.

---

## 4. Model artifact missing

**Symptom.** Tests skip with "Model artifact missing", `make app` refuses to start, or the
app shows "The prediction service is temporarily unavailable."

**Cause.** `make train` has not been run, or the artifacts were deleted.

**Fix.**

```bash
make train
ls -lh deploy/artifacts/
```

Expect five files: `model_pipeline.joblib`, `model_metadata.json`, `feature_schema.json`,
`reference_rates.json`, `model_card.md`.

The application deliberately never trains at startup. A container that trains on boot is
slow, non-deterministic and unauditable.

---

## 5. Streamlit column mismatch

**Symptom**
```
ValueError: The feature names should match those that were passed during fit.
```

**Cause.** The one-row DataFrame does not carry the exact training columns — usually
because `feature_schema.json` and `model_pipeline.joblib` come from different training runs.

**Fix.** Regenerate both together; they are always written by the same run:

```bash
make train
make test    # test_feature_schema_order_matches_training_configuration will confirm
```

Never hand-edit `feature_schema.json`. The app builds every control from it, so an edit
silently changes the model input contract.

---

## 6. Docker build error

**Symptom.** `docker build` fails, or `Cannot connect to the Docker daemon`.

**Causes and fixes.**

| Cause | Fix |
|---|---|
| Daemon not running | Start Docker Desktop and wait for it to report Running, then `docker info` |
| Wrong build context | Build with `deploy/` as the context: `docker build -t churn-app deploy/` |
| Artifacts missing | `make train` before building — `.dockerignore` will not conjure them |
| Slow or failing pip step | Check network; the image installs pinned versions from PyPI |
| Architecture mismatch | On Apple silicon deploying to x86: `docker build --platform linux/amd64 -t churn-app deploy/` |

Verify the built image end to end:

```bash
make docker-run     # builds, runs, and polls /_stcore/health
```

---

## 7. Container starts but the app is unreachable

**Symptom.** The container is running; `http://localhost:7860` does not respond.

**Cause.** Streamlit bound to localhost inside the container, or the port is not published.

**Fix.** The `CMD` must include `--server.address=0.0.0.0`; binding to `127.0.0.1` inside a
container makes it unreachable from the host. Run with `-p 7860:7860`. Check the logs:

```bash
docker logs churn-intelligence-local
```

---

## 8. Git LFS

**Symptom.** GitHub warns about a large file, or the Space checkout produces an LFS pointer
instead of the model.

**Cause.** A model artifact above the size threshold without LFS configured — or LFS
configured but the workflow checking out without it.

**Current status.** `model_pipeline.joblib` is **8.4 KB**, so **LFS is not required**. This
was measured, not assumed. Both workflows already check out with `lfs: true`, so enabling
LFS later needs no workflow change.

**If a future model exceeds ~50 MB:**

```bash
git lfs install
git lfs track "deploy/artifacts/*.joblib"
git add .gitattributes
git commit -m "Track model artifacts with Git LFS"
```

Symptom of the classic mistake: the Space fails to load the model and the file is ~130
bytes of text beginning `version https://git-lfs.github.com/spec/v1`. That is a pointer,
not a model — the checkout did not fetch LFS objects.

---

## 9. Missing `HF_TOKEN`

**Symptom**
```
::error::Actions secret HF_TOKEN is not configured.
```

**Cause.** The secret does not exist, or was created at organisation/environment level
where this workflow cannot see it.

**Fix.** Settings → Secrets and variables → **Actions** → **Secrets** → New repository
secret, named exactly `HF_TOKEN`. Case-sensitive. See `docs/SECURITY.md` §2 for creating a
correctly scoped token.

Also fails if the token exists but lacks **write** permission on the target Space — the
sync then fails with a 403 from Hugging Face rather than at the configuration check.

---

## 10. Missing or malformed `HF_SPACE_ID`

**Symptom**
```
::error::Repository variable HF_SPACE_ID is not configured.
::error::HF_SPACE_ID must be in the form username-or-org/space-name.
```

**Cause.** The variable is absent, or holds a full URL rather than an ID.

**Fix.** Settings → Secrets and variables → Actions → **Variables** tab (not Secrets).

| Correct | Wrong |
|---|---|
| `your-username/customer-churn-intelligence` | `https://huggingface.co/spaces/your-username/customer-churn-intelligence` |
| | `customer-churn-intelligence` (no owner) |
| | `spaces/your-username/customer-churn-intelligence` |

---

## 11. Hugging Face build failure

**Symptom.** The Actions run is green but the Space shows "Build error" or "Runtime error".

**Cause.** The sync succeeded and the *Space* build failed — these are separate steps. This
is exactly why a green workflow must never be reported as a successful deployment.

**Diagnosis.** Open the Space → **Logs** → Build.

| Log message | Fix |
|---|---|
| `no such file or directory: app.py` | The Space root is missing files; confirm `subdirectory: deploy` in the workflow |
| `ModuleNotFoundError` | A dependency missing from `deploy/requirements.txt`; `make test` catches this locally |
| `InconsistentVersionWarning` then a load failure | `scikit-learn` pin does not match the training version; a test enforces this |
| Port binding errors | `app_port: 7860` in the README front matter must match `EXPOSE` and the `CMD` |
| Permission denied on artifacts | The non-root user cannot read them; the Dockerfile's `chmod -R a+rX /app` handles this |
| No `Dockerfile` found | The README front matter must say `sdk: docker` |

Reproduce the Space build locally first — it is the same Dockerfile:

```bash
make docker-run
```

---

## 12. GitHub Actions permission failure

**Symptom.** `Resource not accessible by integration`, or the workflow cannot check out.

**Causes and fixes.**

| Cause | Fix |
|---|---|
| Actions disabled | Settings → Actions → General → allow all actions |
| Workflow permissions too restrictive | These workflows need only `contents: read`, already set |
| Fork pull request | Secrets are not available to fork PRs by design. CI still runs; deployment does not |
| Third-party actions blocked | Settings → Actions → General → allow `actions/*` and `huggingface/*` |

---

## 13. Application shows "temporarily unavailable" **(observed pattern)**

**Symptom.** The generic error message appears instead of a result.

**Cause.** By design the user sees no technical detail. The cause is in the server log.

**Fix.** Read the log — locally it is the terminal running Streamlit; in Docker,
`docker logs <container>`; on a Space, the Runtime logs tab. Look for
`Artifact loading failed` or `Prediction failed for one submitted record`, each followed by
the real traceback.

Most common causes: artifacts missing (§4), or a scikit-learn version mismatch (§11).

---

## 14. Selectboxes or charts look wrong after a Streamlit upgrade **(observed)**

**Symptom.** Inputs render with a light background in dark mode, expander headers show text
like `_arrow_right`, or section cards render empty with the widgets below them.

**Cause.** `deploy/theme.py` styles Streamlit's internals. Three real failures were hit and
fixed during this build:

1. A broad `font-family` rule on `span`/`div` overrode Streamlit's **icon font**, so icon
   ligature names rendered as literal text. The rule is now scoped to `.stApp` only.
2. A hand-written opening `<div>` never enclosed the widgets, because Streamlit closes
   unbalanced HTML at the end of each markdown block. Sections now use
   `st.container(border=True)`.
3. Streamlit followed the operating system's colour scheme and fought the in-app toggle.
   The base theme is now pinned in `deploy/.streamlit/config.toml`.

**Fix after an upgrade.** The overrides target `data-testid` and ARIA attributes rather than
generated class names, which is more stable but not immune. Inspect the element in browser
dev tools and update the selector in `theme.py`. The layout degrades to plain Streamlit
rather than breaking, so this is cosmetic, never functional.

---

## 15. Python version problems

**Symptom.** `bootstrap_macos.sh` reports Python 3.11 was not found, or packages fail to
install.

**Cause.** The default `python3` is a different version — Anaconda's 3.13 is common on
macOS.

**Fix.**

```bash
brew install python@3.11
which -a python3.11
bash scripts/bootstrap_macos.sh
```

The bootstrap script deliberately does not install Homebrew or modify your global Python.
The project targets 3.11 because that is the container runtime; matching them locally means
a local pass predicts a Space pass.

# Security

## 1. Credential model

This repository contains **no credential of any kind**. Nothing needs to be added to it in
order to deploy. Authentication lives entirely in platform settings.

| Item | Type | Where it lives | Who reads it |
|---|---|---|---|
| `HF_TOKEN` | GitHub Actions **secret** | Repository → Settings → Secrets and variables → Actions → **Secrets** | Only the `deploy` job, only as `${{ secrets.HF_TOKEN }}` |
| `HF_SPACE_ID` | GitHub Actions **repository variable** | Same page → **Variables** tab | The `deploy` job, as `${{ vars.HF_SPACE_ID }}` |

`HF_SPACE_ID` is not a secret — it is a public identifier such as
`your-username/customer-churn-intelligence`. It is a variable rather than a secret so it
can be printed in logs for diagnosis. The **token is never printed**, never interpolated
into a URL, and never written to the step summary.

## 2. Creating a least-privilege Hugging Face token

1. Hugging Face → your avatar → **Settings** → **Access Tokens**.
2. **Create new token** → token type **Fine-grained**.
3. Name it for its purpose, e.g. `github-actions-churn-space-deploy`.
4. Grant the **narrowest** scope available:
   - Repository permissions → select **only** the target Space.
   - Permission: **Write** on that Space.
   - Do **not** grant org-wide, all-repositories, inference, or billing scopes.
5. Set an expiry if the account offers one, and diarise the renewal.
6. Copy the value **once** and paste it straight into the GitHub secret. Do not store it in
   a file, a note, a chat message, or a screenshot.

A classic write token also works but grants write access to everything in the account.
Prefer fine-grained; if only classic is available, treat rotation as more urgent.

## 3. Adding the secret and variable in GitHub

```
Repository → Settings → Secrets and variables → Actions
  ├── Secrets tab   → New repository secret
  │     Name:  HF_TOKEN
  │     Value: <paste the token — this is the only time it is visible>
  └── Variables tab → New repository variable
        Name:  HF_SPACE_ID
        Value: username-or-org/space-name
```

GitHub masks secret values in logs, but masking is a safety net, not a control. The
workflow is written so the value is never emitted in the first place.

## 4. Rotation after exposure

If a token is ever pasted into a file, a screenshot, a terminal recording, an issue, or a
chat, treat it as compromised immediately. Rotation order matters:

1. **Revoke first** at Hugging Face → Settings → Access Tokens → delete the token. Revoking
   before cleaning up ends the exposure window immediately.
2. Create a replacement token with the same narrow scope.
3. Update the `HF_TOKEN` secret in GitHub.
4. Remove the value from wherever it leaked.
5. If it was **committed**, note that deleting the file is not enough — the value stays in
   history. Rewrite history (`git filter-repo`, or a fresh repository) and force-push, and
   assume anyone who cloned or forked in the interim still holds it. Revocation in step 1
   is what actually protects you; history rewriting is cleanup.
6. Record the incident and the rotation date in `docs/IMPLEMENTATION_LOG.md` — **without**
   the value.

## 5. Automated secret scanning

`scripts/scan_secrets.sh` runs locally (`make secret-scan`), in CI, and again in the
deployment workflow before anything is pushed.

Detected categories:

| Category | Pattern |
|---|---|
| Hugging Face token | `hf_` followed by 20+ alphanumerics |
| GitHub personal access token | `ghp_` / `gho_` / `ghu_` / `ghs_` / `ghr_` followed by 20+ |
| AWS access key id | `AKIA…` / `ASIA…` + 16 uppercase alphanumerics |
| AWS secret access key | `aws_secret_access_key = …` |
| AWS session token | `aws_session_token = …` |
| Private key | `-----BEGIN … PRIVATE KEY-----` |
| Bearer token | `Authorization: Bearer …` (20+ chars) |

It also fails on the mere presence of credential-bearing filenames: `.env`, `credentials`,
`credentials.json`, `.hf_token`, `id_rsa`, `id_ed25519`.

**The scanner prints file paths and categories only — never the matched text.** A scanner
that echoed its findings would leak the secret into CI logs, which are often more widely
readable than the repository.

`tests/test_deployment_files.py` repeats the scan inside pytest, so `make test` alone
catches a leak even if the shell script is skipped.

## 6. Version-control hygiene

`.gitignore` covers virtual environments, Python caches, `.env` files (except
`.env.example`), `*.pem`, `*.key`, `id_rsa*`, `id_ed25519*`, `.aws/`, `credentials*`,
`.hf_token`, `.huggingface/`, logs, local screenshots and editor directories.

Before every commit:

```bash
make test
make secret-scan
git diff
git diff --cached --name-only
```

The last command is the one that matters: it shows exactly what is about to be committed.
`scripts/verify_release.sh` additionally fails if any tracked file matches a
credential-bearing filename.

## 7. Public-repository implications

If the repository is public, assume everything in it is permanently public:

- Every commit in history is readable, including files deleted later.
- Automated scrapers harvest committed tokens within minutes. Rotation speed, not deletion
  speed, is what limits the damage.
- The dataset is Apache-2.0 licensed and safe to publish. It contains no personal data —
  it is a **fictional** sample.
- The model artifact is derived from that sample and carries no confidential information.
- Screenshots in the report can leak more than code. Check every screenshot for tokens,
  private URLs, email addresses, browser bookmarks and unrelated open tabs before
  including it.

## 8. Container security

| Control | Implementation |
|---|---|
| Non-root execution | `useradd --uid 1000 appuser`; `USER appuser` before `CMD` |
| Minimal base | `python:3.11-slim` |
| No secrets in the image | No `ARG`/`ENV` carries a credential; the Dockerfile is tested for token markers |
| Minimal contents | Only `app.py`, `theme.py`, `charts.py`, `.streamlit/`, `artifacts/` and pinned requirements are copied; `.dockerignore` excludes the rest |
| No build-time network calls beyond pip | Nothing is fetched from a CDN at build or request time |
| Read-only artifacts | `chmod -R a+rX /app`; the application only reads them |
| Writable cache confined | `MPLCONFIGDIR=/tmp/matplotlib`, owned by the app user |
| Health endpoint | `HEALTHCHECK` against `/_stcore/health` |
| XSRF protection | Enabled in `.streamlit/config.toml` |

## 9. Application-level controls

- **No stack traces reach the user.** `showErrorDetails = false` in the Streamlit config,
  and both failure paths in `app.py` log the exception server-side while showing a generic
  message with no path or identifier.
- **No user input is interpolated into HTML.** Every value rendered into the result card
  passes through `html.escape`, and the only interpolated values are numbers and strings
  from a fixed internal set.
- **No external resources.** No CDN font, stylesheet, script or image — the page is fully
  self-contained, which removes an entire supply-chain and privacy surface.
- **No data is collected or transmitted.** `gatherUsageStats = false`. Inputs are scored in
  memory and never persisted; the server log records the resulting probability, not the
  customer attributes.

## 10. Dependency and artifact considerations

- Runtime dependencies are pinned exactly, so a rebuild installs the same versions that
  were tested.
- `scikit-learn` is pinned to the training version, enforced by a test. Loading a pickled
  estimator under a different minor version is unsupported.
- `joblib.load` executes pickled objects and must only ever be pointed at an artifact this
  project produced. Never load a `.joblib` file from an untrusted source.
- The artifact is regenerated by `make train` from the verified dataset, so its provenance
  is reproducible rather than assumed.
- No dependency-vulnerability scanner is wired into CI. Adding `pip-audit` to the CI job is
  the recommended next step and is listed as a future enhancement.

## 11. Known gaps

Recorded honestly rather than omitted:

- No automated dependency-vulnerability scanning (see above).
- No container image scanning (Trivy or equivalent).
- No signed commits or signed container images.
- The secret scanner uses pattern matching; a credential in an unusual format could evade
  it. It reduces risk, it does not eliminate it.
- No fairness audit has been performed on the model, which is a governance gap as well as
  an ethical one. See `deploy/artifacts/model_card.md`.

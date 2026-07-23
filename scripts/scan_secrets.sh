#!/usr/bin/env bash
# Scan project files for credential patterns.
#
# Only the file path and the pattern category are printed. The matched text is
# never echoed, so running this in CI cannot itself leak a secret.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Scanning for credential patterns in ${ROOT_DIR}"
echo "------------------------------------------------------------------"

# Categories and their detection patterns (extended regular expressions).
#
# The AWS assignment patterns require a credential-shaped value (16+ base64
# characters) rather than any non-space token. Matching anything would also flag
# the documentation of these patterns in docs/SECURITY.md, and a scanner that
# cries wolf on its own documentation gets ignored — a worse outcome than a
# slightly narrower pattern. A real AWS secret key is 40 base64 characters.
CATEGORIES=(
  "hugging_face_token|hf_[A-Za-z0-9]{20,}"
  "github_personal_access_token|gh[pousr]_[A-Za-z0-9]{20,}"
  "aws_access_key_id|(AKIA|ASIA)[0-9A-Z]{16}"
  "aws_secret_access_key|aws_secret_access_key[[:space:]]*[=:][[:space:]]*[\"']?[A-Za-z0-9/+=]{16,}"
  "aws_session_token|aws_session_token[[:space:]]*[=:][[:space:]]*[\"']?[A-Za-z0-9/+=]{16,}"
  "private_key_header|-----BEGIN( (RSA|EC|OPENSSH|PGP))? PRIVATE KEY-----"
  "generic_bearer_token|[Aa]uthorization:[[:space:]]*[Bb]earer[[:space:]]+[A-Za-z0-9._-]{20,}"
)

EXCLUDE_DIRS=(--exclude-dir=.git --exclude-dir=.venv --exclude-dir=venv
              --exclude-dir=__pycache__ --exclude-dir=.pytest_cache
              --exclude-dir=node_modules --exclude-dir=.ipynb_checkpoints)

# This script contains the patterns themselves, so it must not scan itself.
EXCLUDE_FILES=(--exclude=scan_secrets.sh --exclude=test_deployment_files.py)

findings=0

for entry in "${CATEGORIES[@]}"; do
  category="${entry%%|*}"
  pattern="${entry#*|}"

  # -l prints file names only: the matched text never reaches stdout.
  if matches="$(grep -rlE "${pattern}" . "${EXCLUDE_DIRS[@]}" "${EXCLUDE_FILES[@]}" 2>/dev/null)"; then
    if [ -n "$matches" ]; then
      while IFS= read -r file; do
        [ -z "$file" ] && continue
        echo "FINDING: ${file#./} matched pattern category '${category}'"
        findings=$((findings + 1))
      done <<< "$matches"
    fi
  fi
done

# Credential-bearing files that must never be tracked, regardless of content.
FORBIDDEN_FILES=(".env" "credentials" "credentials.json" ".hf_token" "id_rsa" "id_ed25519")
for name in "${FORBIDDEN_FILES[@]}"; do
  while IFS= read -r found; do
    [ -z "$found" ] && continue
    case "$found" in
      *"/.venv/"*|*"/.git/"*) continue ;;
    esac
    echo "FINDING: ${found#./} is a credential-bearing filename and must not exist here"
    findings=$((findings + 1))
  done < <(find . -name "$name" -not -path "./.git/*" -not -path "./.venv/*" 2>/dev/null)
done

echo "------------------------------------------------------------------"
if [ "$findings" -ne 0 ]; then
  echo "SECRET SCAN FAILED: ${findings} finding(s)." >&2
  echo "Remove the credential, rotate it immediately at the provider, and" >&2
  echo "rewrite history if it was ever committed." >&2
  exit 1
fi

echo "SECRET SCAN PASSED: no credential patterns detected."

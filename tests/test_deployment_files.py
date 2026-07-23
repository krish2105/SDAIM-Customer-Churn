"""Deployment packaging, workflow configuration and secret-hygiene tests.

These run in CI before anything is pushed to the Hugging Face Space, so a
misconfigured Dockerfile, a missing runtime dependency or a leaked credential
fails the build rather than the deployment.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest

from src import config

DEPLOY_DIR = config.DEPLOY_DIR
WORKFLOWS_DIR = config.PROJECT_ROOT / ".github" / "workflows"

RUNTIME_MODULES = {"app.py", "theme.py", "charts.py"}

#: Import name -> distribution name, for imports whose package name differs.
IMPORT_TO_DISTRIBUTION = {
    "sklearn": "scikit-learn",
    "PIL": "pillow",
}

#: Modules in the standard library or provided by the runtime itself.
STANDARD_MODULES = {
    "json",
    "logging",
    "pathlib",
    "typing",
    "html",
    "__future__",
    "dataclasses",
    "datetime",
    "os",
    "sys",
    "re",
    "math",
}


def _local_module_names() -> set[str]:
    return {path.stem for path in DEPLOY_DIR.glob("*.py")}


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module.split(".")[0])
    return modules


# --------------------------------------------------------------------------
# Runtime files
# --------------------------------------------------------------------------


def test_docker_runtime_files_exist() -> None:
    for name in ("Dockerfile", "requirements.txt", "README.md", ".dockerignore", *RUNTIME_MODULES):
        assert (DEPLOY_DIR / name).is_file(), f"deploy/{name} is missing"


def test_artifacts_are_present_for_deployment() -> None:
    for path in (
        config.MODEL_PATH,
        config.METADATA_PATH,
        config.FEATURE_SCHEMA_PATH,
        config.MODEL_CARD_PATH,
    ):
        assert path.is_file(), f"{path.name} is missing — run `make train`"


def test_app_never_imports_the_training_package() -> None:
    """The Space image contains no training code, so app.py must not need it."""
    for name in RUNTIME_MODULES:
        imports = _top_level_imports(DEPLOY_DIR / name)
        assert "src" not in imports, f"deploy/{name} must not import the src package"


def test_deployment_requirements_cover_every_runtime_import() -> None:
    requirements = (DEPLOY_DIR / "requirements.txt").read_text(encoding="utf-8")
    declared = {
        re.split(r"[=<>!~\[]", line.strip())[0].lower()
        for line in requirements.splitlines()
        if line.strip() and not line.startswith("#")
    }
    local = _local_module_names()

    for name in RUNTIME_MODULES:
        for module in _top_level_imports(DEPLOY_DIR / name):
            if module in STANDARD_MODULES or module in local:
                continue
            distribution = IMPORT_TO_DISTRIBUTION.get(module, module).lower()
            assert distribution in declared, (
                f"deploy/{name} imports '{module}' but deploy/requirements.txt "
                f"does not declare '{distribution}'"
            )


def test_deployment_requirements_are_pinned() -> None:
    for line in (DEPLOY_DIR / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            assert "==" in line, f"Runtime dependency '{line}' must be pinned"


def test_runtime_scikit_learn_pin_matches_training_environment(
    model_metadata: dict[str, Any],
) -> None:
    """Unpickling under a different scikit-learn minor version is unsupported."""
    trained_with = model_metadata["environment"]["scikit_learn"]
    requirements = (DEPLOY_DIR / "requirements.txt").read_text(encoding="utf-8")
    assert f"scikit-learn=={trained_with}" in requirements, (
        f"Artifact was trained with scikit-learn {trained_with}; pin the same version."
    )


# --------------------------------------------------------------------------
# Dockerfile and Space configuration
# --------------------------------------------------------------------------


def test_dockerfile_configuration() -> None:
    content = (DEPLOY_DIR / "Dockerfile").read_text(encoding="utf-8")
    assert "python:3.11-slim" in content
    assert "PYTHONDONTWRITEBYTECODE=1" in content
    assert "PYTHONUNBUFFERED=1" in content
    assert "PIP_NO_CACHE_DIR=1" in content
    assert "useradd" in content, "A non-root application user is required"
    assert "USER appuser" in content
    assert "WORKDIR /app" in content
    assert "EXPOSE 7860" in content
    assert "--server.port=7860" in content
    assert "--server.address=0.0.0.0" in content
    assert "--server.headless=true" in content


def test_dockerfile_contains_no_credentials() -> None:
    content = (DEPLOY_DIR / "Dockerfile").read_text(encoding="utf-8")
    for marker in ("hf_", "HF_TOKEN", "AWS_SECRET", "PASSWORD", "api_key"):
        assert marker not in content, f"Dockerfile must not reference {marker}"


def test_space_readme_metadata_uses_docker_and_port_7860() -> None:
    lines = (DEPLOY_DIR / "README.md").read_text(encoding="utf-8").splitlines()
    assert lines[0].strip() == "---", "Space README must open with a YAML front-matter block"
    closing = lines[1:].index("---") + 1
    front_matter = "\n".join(lines[1:closing])
    assert "sdk: docker" in front_matter
    assert "app_port: 7860" in front_matter
    assert "title:" in front_matter
    assert "emoji:" in front_matter


# --------------------------------------------------------------------------
# Workflows
# --------------------------------------------------------------------------


def test_workflows_exist() -> None:
    assert (WORKFLOWS_DIR / "ci.yml").is_file()
    assert (WORKFLOWS_DIR / "deploy.yml").is_file()


def test_deploy_workflow_reads_only_the_expected_secret_and_variable() -> None:
    content = (WORKFLOWS_DIR / "deploy.yml").read_text(encoding="utf-8")
    assert "secrets.HF_TOKEN" in content
    assert "vars.HF_SPACE_ID" in content

    referenced_secrets = set(re.findall(r"secrets\.([A-Z_]+)", content))
    assert referenced_secrets <= {"HF_TOKEN", "GITHUB_TOKEN"}, (
        f"Unexpected secrets referenced: {referenced_secrets}"
    )
    referenced_vars = set(re.findall(r"vars\.([A-Z_]+)", content))
    assert referenced_vars <= {"HF_SPACE_ID"}, f"Unexpected variables: {referenced_vars}"


def test_deploy_workflow_configuration() -> None:
    content = (WORKFLOWS_DIR / "deploy.yml").read_text(encoding="utf-8")
    assert "repo_type: space" in content
    assert "space_sdk: docker" in content
    assert "subdirectory: deploy" in content
    assert "concurrency:" in content
    assert "workflow_dispatch:" in content
    assert "contents: read" in content
    assert "lfs: true" in content


def test_workflows_never_echo_the_token() -> None:
    for workflow in WORKFLOWS_DIR.glob("*.yml"):
        content = workflow.read_text(encoding="utf-8")
        for line in content.splitlines():
            stripped = line.strip()
            if "secrets.HF_TOKEN" in stripped:
                assert not stripped.startswith(("echo", "- run: echo", "run: echo")), (
                    f"{workflow.name} must never echo the token"
                )
                assert "https://" not in stripped, (
                    f"{workflow.name} must not interpolate the token into a URL"
                )


# --------------------------------------------------------------------------
# Secret hygiene across the project
# --------------------------------------------------------------------------

# The AWS assignment patterns require a credential-shaped value rather than any
# non-space token. Matching `\S+` would also flag the documentation of these very
# patterns in docs/SECURITY.md, and a scanner that cries wolf on its own docs
# gets muted — which is worse than a slightly narrower pattern. A real AWS secret
# is 40 base64 characters; the session token is far longer.
SECRET_PATTERNS = {
    "hugging_face_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    "github_pat": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "aws_secret_key": re.compile(
        r"aws_secret_access_key\s*[=:]\s*[\"']?[A-Za-z0-9/+=]{16,}", re.IGNORECASE
    ),
    "aws_session_token": re.compile(
        r"aws_session_token\s*[=:]\s*[\"']?[A-Za-z0-9/+=]{16,}", re.IGNORECASE
    ),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
}

SKIP_DIRECTORIES = {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules", ".ipynb_checkpoints"}
SCANNED_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".json", ".toml", ".sh", ".txt", ".cfg", ".ipynb", ""}


def _scannable_files() -> list[Path]:
    files: list[Path] = []
    for path in config.PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRECTORIES for part in path.parts):
            continue
        if path.suffix.lower() not in SCANNED_SUFFIXES:
            continue
        if path.stat().st_size > 2_000_000:
            continue
        files.append(path)
    return files


def test_no_secret_patterns_in_project_files() -> None:
    findings: list[str] = []
    for path in _scannable_files():
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:  # pragma: no cover - unreadable file
            continue
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                # Report the location and category only, never the match itself.
                findings.append(f"{path.relative_to(config.PROJECT_ROOT)}: {name}")
    assert not findings, f"Potential secrets detected: {findings}"


def test_gitignore_covers_sensitive_paths() -> None:
    gitignore = (config.PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    for entry in (".venv", "__pycache__", ".env", "*.pem", ".aws/", "*.log"):
        assert entry in gitignore, f".gitignore should cover {entry}"


def test_no_env_file_is_committed_to_the_repository() -> None:
    for path in config.PROJECT_ROOT.rglob(".env"):
        if ".venv" in path.parts:
            continue
        pytest.fail(f"A .env file exists at {path}; it must never be committed.")

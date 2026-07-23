# Runtime image for the Hugging Face Docker Space.
# Build context is the deploy/ directory:
#   docker build -t churn-app deploy/
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Non-root application user. Hugging Face Spaces run containers as uid 1000.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# Dependencies first so the layer is cached independently of application code.
COPY --chown=appuser:appuser requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/requirements.txt

# Only the runtime files. Training code, raw data, notebooks, tests and reports
# are deliberately excluded — see .dockerignore.
COPY --chown=appuser:appuser app.py theme.py charts.py explain.py batch.py rationale.py /app/
COPY --chown=appuser:appuser .streamlit /app/.streamlit
COPY --chown=appuser:appuser artifacts /app/artifacts

# Read access to the artifacts for the non-root user, and a writable cache dir.
RUN chmod -R a+rX /app && mkdir -p /tmp/matplotlib && chown appuser:appuser /tmp/matplotlib

USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/_stcore/health', timeout=4).status==200 else 1)"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true"]

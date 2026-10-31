# syntax=docker/dockerfile:1
# Isolated runtime for the deficit pipeline. No Windows .exe launchers, no host Python needed.
#   docker compose build
#   docker compose run --rm deficit run --synthetic
FROM python:3.12-slim

# Optional extras from pyproject.toml. "lstm" pulls TensorFlow (large). Default: none. For LSTM:
#   $env:EXTRAS="lstm"; docker compose build
ARG EXTRAS=

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MPLBACKEND=Agg

WORKDIR /app

# Layers 1-2: third-party dependencies only (cached until pyproject.toml changes).
# Core and extras are installed in separate layers, so a failed or corrupted download of the
# large optional packages (TensorFlow) does not throw away the finished core layer.
COPY pyproject.toml ./
RUN python - <<'EOF'
import os, tomllib
p = tomllib.load(open("pyproject.toml", "rb"))["project"]
extra = []
for name in filter(None, os.environ.get("EXTRAS", "").split(",")):
    extra += p["optional-dependencies"][name]
open("/tmp/req-core.txt", "w").write("\n".join(p["dependencies"] + ["pytest"]) + "\n")
open("/tmp/req-extra.txt", "w").write("\n".join(extra) + ("\n" if extra else ""))
EOF
# --retries/--timeout: long downloads over a slow connection are the usual failure here.
RUN pip install --retries 10 --timeout 120 -r /tmp/req-core.txt
RUN if [ -s /tmp/req-extra.txt ]; then \
        pip install --retries 10 --timeout 120 -r /tmp/req-extra.txt; \
    fi

# entsoe-py uses the legacy zone name Europe/Kiev, which Debian 13 moved out of the default tzdata package.
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends tzdata-legacy \
    && rm -rf /var/lib/apt/lists/*

# Layer 2: the project itself. Editable, so PROJECT_ROOT resolves to /app (config/, data/ ...).
COPY README.md ./
COPY src ./src
COPY config ./config
COPY tests ./tests
RUN pip install --no-deps -e .

# Run as an unprivileged user; output directories are bind-mounted from the host.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p data artifacts reports \
    && chown -R app:app /app
USER app

ENTRYPOINT ["deficit"]
CMD ["--help"]

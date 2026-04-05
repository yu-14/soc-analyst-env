# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

ARG ENV_NAME=soc_analyst_env

COPY . /app/env

WORKDIR /app/env/soc_analyst_env

RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

FROM ${BASE_IMAGE}

WORKDIR /app

COPY --from=builder /app/env/soc_analyst_env/.venv /app/.venv
COPY --from=builder /app/env /app/env

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env/soc_analyst_env:$PYTHONPATH"

EXPOSE 7860

CMD ["sh", "-c", "cd /app/env/soc_analyst_env && uvicorn server.app:app --host 0.0.0.0 --port 7860"]
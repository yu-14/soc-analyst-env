# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause

"""FastAPI app exposing SocAnalystEnvironment (HTTP + WebSocket)."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv-core is required. Install dependencies with: uv sync"
    ) from e

try:
    from models import SocAction, SocObservation
    from server.soc_analyst_environment import SocAnalystEnvironment
except ImportError:
    from ..models import SocAction, SocObservation
    from .soc_analyst_environment import SocAnalystEnvironment

app = create_app(
    SocAnalystEnvironment,
    SocAction,
    SocObservation,
    env_name="soc_analyst_env",
    max_concurrent_envs=4,
)


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

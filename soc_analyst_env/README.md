# soc_analyst_env (OpenEnv)

Synthetic SOC analyst triage: bounded logs, MITRE ATT&CK-style technique IDs, and deterministic graders (`easy` / `medium` / `hard`).

See the repository root [README.md](../README.md) for full documentation, inference, and Docker usage.

Quick check:

```bash
uv sync
uv run openenv validate --verbose
```

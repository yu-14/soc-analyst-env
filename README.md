# Round_One: OpenEnv SOC Analyst Environment

Synthetic **security operations (SOC)** triage for agent evaluation: parse bounded network and application logs, map findings to [MITRE ATT&CK](https://attack.mitre.org/) technique IDs (synthetic scenarios only), and propose ordered remediation steps. The domain targets **alert fatigue** and multi-source correlation—common enterprise pain points—without using real customer data.

## Repository layout

| Path | Purpose |
|------|---------|
| [soc_analyst_env/](soc_analyst_env/) | OpenEnv package (`openenv validate`, Docker, HF Space) |
| [soc_analyst_env/openenv.yaml](soc_analyst_env/openenv.yaml) | OpenEnv manifest |
| [soc_analyst_env/models.py](soc_analyst_env/models.py) | Pydantic `SocAction`, `SocObservation`, `SocState`, `SocReward` |
| [soc_analyst_env/graders.py](soc_analyst_env/graders.py) | Deterministic scores in `[0.0, 1.0]` |
| [soc_analyst_env/tasks.py](soc_analyst_env/tasks.py) | Registered task list + grader bindings |
| [soc_analyst_env/data/](soc_analyst_env/data/) | Gold JSON per task (`identify_malicious_ip`, …) |
| [inference.py](inference.py) | OpenAI-compatible client + **strict stdout** logs |

## Action space (`SocAction`)

All actions extend OpenEnv `Action` with a required `kind`:

- `noop` — no operational effect; small negative reward (discourages spam).
- `submit_hypothesis` — provisional `technique_ids` (+ optional `rationale_tag`); partial overlap with gold techniques increases shaped score (capped).
- `submit_correlation` — `service_a`, `service_b`, `link_key`, `link_value` (medium task); partial credit vs gold correlation.
- `finalize_triage` — **required to finish**: `verdict` (`true_positive` \| `false_positive` \| `benign`), `primary_technique`, `technique_ids`, `remediation_steps`, and optionally `correlation_*` fields on finalize.
- `destructive_network_block` — wide blocking action; large negative reward.

## Observation space (`SocObservation`)

- `task`, `instruction`, `alert_id`, `alert_rule`, `alert_severity`
- `log_view` — newline-separated synthetic lines (includes noise on **hard**)
- `max_steps`, `available_commands`, `feedback`
- `done`, `reward` (OpenEnv)
- `final_grader_score`, `episode_success` (populated when the episode ends; success if grader ≥ **0.85**)

## Reward model (`SocReward`)

Structured breakdown is attached on the server for logging; the scalar passed over the wire is `Observation.reward`. Shaped updates accumulate in `shaped_score`; terminal `finalize_triage` blends **0.5 × shaped + 0.5 × grader**.

## Tasks (three registered graders)

Each task has a **canonical id**, a **JSON gold file** in `data/`, and a **`def grade_*` function** in `graders.py`. The same definitions are listed under `tasks:` in [openenv.yaml](soc_analyst_env/openenv.yaml) and in `SOC_ANALYST_TASKS` / `GRADERS_BY_TASK_ID` in [tasks.py](soc_analyst_env/tasks.py).

| Canonical id | Legacy alias | SOC focus | Grader function | Agent hints (on `finalize_triage`) |
|--------------|--------------|-----------|-----------------|-------------------------------------|
| `identify_malicious_ip` | `easy` | Malicious source IP from auth logs | `grade_identify_malicious_ip` | Set `metadata.malicious_ip` to the IPv4 (e.g. `203.0.113.50`). |
| `find_compromised_account` | `medium` | Compromised account + service correlation | `grade_find_compromised_account` | Set `metadata.compromised_account` (e.g. `anonymous`); use `submit_correlation`. |
| `recommend_firewall_rule` | `hard` | Noisy APT + firewall/WAF playbook steps | `grade_recommend_firewall_rule` | `remediation_steps` must include each id in `gold_firewall_rules` (see JSON). |

Reset:

```python
client.reset(task="identify_malicious_ip")
# or legacy aliases:
client.reset(task="easy")    # same as identify_malicious_ip
client.reset(task="medium")  # find_compromised_account
client.reset(task="hard")    # recommend_firewall_rule
```

## Setup

**Environment server** (from [soc_analyst_env/](soc_analyst_env/)):

```bash
cd soc_analyst_env
uv sync
uv run server
# or: uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

**Validation**

```bash
cd soc_analyst_env
uv run openenv validate --verbose
# With a running server:
uv run openenv validate --url http://127.0.0.1:8000 --verbose
```

**Docker** (build context = `soc_analyst_env/`):

```bash
cd soc_analyst_env
docker build -f server/Dockerfile -t soc-analyst-env:latest .
docker run --rm -p 8000:8000 soc-analyst-env:latest
```

Default OpenEnv mode is **simulation**, so `POST /reset`, `POST /step`, and `GET /state` are registered (required for local checks and typical HF Space configs).

## Inference script (`inference.py`)

Install the env and OpenAI client (from repo root):

```bash
pip install openai
pip install -e ./soc_analyst_env
```

Environment variables:

| Variable | Role |
|----------|------|
| `API_BASE_URL` | OpenAI-compatible API base (e.g. `https://api.openai.com/v1` or an HF router) |
| `MODEL_NAME` | Model id |
| `HF_TOKEN` | API key (or set `OPENAI_API_KEY`) |
| `OPENENV_BASE_URL` | Full URL of the OpenEnv server (overrides host/port) |
| `OPENENV_HOST` / `OPENENV_PORT` | Default host `127.0.0.1`, port **`7860`** (HF Spaces / many Dockerfiles); `PORT` is used if `OPENENV_PORT` is unset |
| `ENV_WAIT_TIMEOUT` | Seconds to poll `/health` before failing (default **60**) |
| `TASK` | `easy` \| `medium` \| `hard` (or pass `--task`) |
| `BENCHMARK` | Label for `[START]` line (default `soc_analyst_env`) |
| `MAX_LLM_STEPS` | Upper bound on LLM turns |

Strict stdout format (two decimal rewards; lowercase booleans):

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
```

Example:

```bash
# Default without OPENENV_BASE_URL is http://127.0.0.1:7860 (see also PORT / OPENENV_PORT)
export OPENENV_BASE_URL=http://127.0.0.1:8000   # optional: local OpenEnv on 8000
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=...
python inference.py --task medium
```

## Baseline scores (reference)

Run `inference.py` with your model and fill in empirical results.

| Task | Oracle (direct correct `finalize_triage`) | Notes |
|------|-------------------------------------------|--------|
| easy | ~1.00 grader | `false_positive`, empty techniques |
| medium | ~1.00 grader | Correct `trace_id` correlation + techniques |
| hard | ~1.00 grader | Five techniques + five remediation steps in order |

Shaped **episode reward** on oracle finalization is typically **~0.75** on easy (0.5×0.5 shaped + 0.5×1.0 grader) unless hypotheses/correlation steps increase `shaped_score` first.

## License

BSD-3-Clause (header per Meta OpenEnv template).

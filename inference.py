#!/usr/bin/env python3
"""
LLM-driven rollouts for the SOC analyst OpenEnv environment.

Runs ALL 3 registered tasks in a single invocation so the hackathon
validator counts 3 graded episodes from the stdout logs.

Env vars (hackathon-mandated):
  API_BASE_URL  – LLM endpoint          (default: https://api.openai.com/v1)
  MODEL_NAME    – model id              (default: gpt-4o-mini)
  HF_TOKEN      – API key               (REQUIRED, no default)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, List, Optional

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from openai import OpenAI

from soc_analyst_env.client import SocAnalystEnv
from soc_analyst_env.models import SocAction

ALL_TASKS: List[str] = [
    "identify_malicious_ip",
    "find_compromised_account",
    "recommend_firewall_rule",
]

BENCHMARK = "soc_analyst_env"
MAX_LLM_STEPS = 5


# ---------------------------------------------------------------------------
# BULLETPROOF reward clamping — nothing outside (0, 1) ever reaches stdout
# ---------------------------------------------------------------------------

def safe_reward(r: Any) -> float:
    """Clamp ANY value to strictly (0, 1) exclusive. Handles None, NaN, Inf."""
    try:
        v = float(r) if r is not None else 0.01
    except (TypeError, ValueError):
        v = 0.01
    if v != v:  # NaN check
        v = 0.01
    return max(0.01, min(0.99, v))


def fmt_reward(r: Any) -> str:
    """Format a reward value to 2 decimal places, guaranteed in [0.01, 0.99]."""
    return f"{safe_reward(r):.2f}"


def fmt_bool(b: bool) -> str:
    return "true" if b else "false"


def fmt_error(msg: Optional[str]) -> str:
    if msg is None:
        return "null"
    s = str(msg).replace("\n", " ").replace("\r", " ").strip()
    return s[:240] if len(s) <= 240 else s[:237] + "..."


def compact_action(action: SocAction) -> str:
    return json.dumps(action.model_dump(), separators=(",", ":"))


NOOP_STR = json.dumps({"kind": "noop", "metadata": {}}, separators=(",", ":"))


def fmt_rewards_list(rewards: List[float]) -> str:
    """Format the entire rewards list for [END] line, every element clamped."""
    if not rewards:
        return "0.01"
    return ",".join(fmt_reward(r) for r in rewards)


# ---------------------------------------------------------------------------
# Logging helpers — single point of truth for [START], [STEP], [END]
# ---------------------------------------------------------------------------

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action_str: str, reward: Any, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={fmt_reward(reward)} done={fmt_bool(done)} error={fmt_error(error)}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(
        f"[END] success={fmt_bool(success)} steps={steps} score={safe_reward(score):.3f} rewards={fmt_rewards_list(rewards)}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Wait for the FastAPI container to become healthy
# ---------------------------------------------------------------------------

def _resolve_env_url() -> str:
    explicit = os.environ.get("OPENENV_BASE_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    host = os.environ.get("OPENENV_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = (
        os.environ.get("OPENENV_PORT", "").strip()
        or os.environ.get("PORT", "").strip()
        or "7860"
    )
    return f"http://{host}:{port}"


def wait_for_env_ready(base_url: str, timeout_s: float = 120.0) -> bool:
    health = f"{base_url.rstrip('/')}/health"
    deadline = time.monotonic() + timeout_s
    delay = 1.0
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(health, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(delay, remaining))
        delay = min(delay * 1.5, 8.0)
    return False


# ---------------------------------------------------------------------------
# Extract JSON from LLM output (may be wrapped in markdown fences)
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    m2 = re.search(r"\{[\s\S]*\}\s*$", text)
    if m2:
        text = m2.group(0)
    return json.loads(text)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a SOC analyst agent. You receive alert context and logs. \
Respond with a SINGLE JSON object (no markdown, no extra text) matching:

{
  "kind": "finalize_triage",
  "verdict": "true_positive" | "false_positive" | "benign",
  "technique_ids": ["T1190", ...],
  "primary_technique": "T1190" or null,
  "remediation_steps": ["step_one", "step_two"],
  "service_a": "", "service_b": "", "link_key": "", "link_value": "",
  "correlation_service_a": "", "correlation_service_b": "",
  "correlation_link_key": "", "correlation_link_value": "",
  "metadata": {},
  "rationale_tag": ""
}

Rules:
- For the identify_malicious_ip task: set metadata.malicious_ip to the attacking IPv4.
- For find_compromised_account: use submit_correlation first (service_a, service_b, \
link_key, link_value), then finalize with metadata.compromised_account.
- For recommend_firewall_rule: include all MITRE technique IDs and ordered \
remediation_steps using lowercase_snake_case ids.
- If a task looks benign or false_positive, say so and leave technique_ids empty.
- Always finalize with kind=finalize_triage and a verdict."""


# ---------------------------------------------------------------------------
# Run one episode for a given task
# ---------------------------------------------------------------------------

def run_task(
    task_name: str,
    model: str,
    llm: OpenAI,
    env_url: str,
) -> None:
    rewards: List[float] = []
    steps = 0
    success = False

    log_start(task_name, model)

    try:
        with SocAnalystEnv(base_url=env_url).sync() as env:
            result = env.reset(task=task_name)
            obs = result.observation

            for _ in range(MAX_LLM_STEPS):
                if result.done:
                    break

                user_msg = json.dumps({
                    "task": task_name,
                    "instruction": obs.instruction,
                    "alert": {
                        "id": obs.alert_id,
                        "rule": obs.alert_rule,
                        "severity": obs.alert_severity,
                    },
                    "log_view": obs.log_view,
                    "feedback": obs.feedback,
                    "available_commands": obs.available_commands,
                    "step": steps + 1,
                    "max_steps": obs.max_steps,
                }, ensure_ascii=False)

                action: SocAction
                step_err: Optional[str] = None
                try:
                    resp = llm.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0.1,
                    )
                    raw = resp.choices[0].message.content or "{}"
                    data = _extract_json(raw)
                    if "metadata" not in data:
                        data["metadata"] = {}
                    action = SocAction.model_validate(data)
                except Exception as e:
                    action = SocAction(kind="noop", metadata={})
                    step_err = f"llm_parse: {e}"

                try:
                    result = env.step(action)
                    obs = result.observation
                    rw = safe_reward(result.reward if result.reward is not None else obs.reward)
                except Exception as e:
                    steps += 1
                    rewards.append(safe_reward(0))
                    log_step(steps, compact_action(action), 0, True, f"env.step: {e}")
                    break

                steps += 1
                rewards.append(rw)
                done_now = result.done

                if obs.final_grader_score is not None and obs.final_grader_score >= 0.85:
                    success = True
                if obs.episode_success:
                    success = True

                log_step(steps, compact_action(action), rw, done_now, step_err)

                if done_now:
                    break

            if not result.done:
                action = SocAction(
                    kind="finalize_triage",
                    verdict="benign",
                    technique_ids=[],
                    remediation_steps=[],
                    metadata={},
                )
                try:
                    result = env.step(action)
                    obs = result.observation
                    rw = safe_reward(result.reward if result.reward is not None else obs.reward)
                    steps += 1
                    rewards.append(rw)
                    if obs.episode_success:
                        success = True
                    log_step(steps, compact_action(action), rw, result.done, None)
                except Exception as e:
                    steps += 1
                    rewards.append(safe_reward(0))
                    log_step(steps, compact_action(action), 0, True, str(e))

    except Exception as e:
        if steps == 0:
            steps = 1
            rewards.append(safe_reward(0))
            log_step(1, NOOP_STR, 0, True, f"connection: {e}")

    if not rewards:
        rewards.append(safe_reward(0))

    final_score = safe_reward(rewards[-1])
    log_end(success, steps, final_score, rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    api_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        for t in ALL_TASKS:
            log_start(t, model_name)
            log_step(1, NOOP_STR, 0, True, "HF_TOKEN not set")
            log_end(False, 1, safe_reward(0), [safe_reward(0)])
        sys.exit(0)

    env_url = _resolve_env_url()

    if not wait_for_env_ready(env_url, timeout_s=120):
        for t in ALL_TASKS:
            log_start(t, model_name)
            log_step(1, NOOP_STR, 0, True, "env not reachable")
            log_end(False, 1, safe_reward(0), [safe_reward(0)])
        sys.exit(0)

    llm = OpenAI(base_url=api_base_url, api_key=hf_token)

    for task_name in ALL_TASKS:
        try:
            run_task(task_name, model_name, llm, env_url)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            log_start(task_name, model_name)
            log_step(1, NOOP_STR, 0, True, "fatal")
            log_end(False, 1, safe_reward(0), [safe_reward(0)])


if __name__ == "__main__":
    main()

"""Everything you might need to change lives here: project, models, judge.

Capital One edition. Two things differ from the public version of this repo,
and nothing else does:

  1. Weave comes from Capital One's internal wrapper, `c1_aiml_aem`.
  2. Every model call goes through Capital One's internal inference gateway
     (`c1.aiml.genai.inference`), so there are no provider API keys. The three
     candidates are three models served by FM Hosting, not three vendors.

All values come from environment variables (see .env.example), so nothing
secret is ever written into the repo.
"""

from __future__ import annotations

import datetime as _dt
import os

# ---------------------------------------------------------------------------
# Where runs land in Weave:  <WANDB_BASE_URL>/<WANDB_ENTITY>/<WANDB_PROJECT>/weave
# Defaults to the Capital One instance; override WANDB_BASE_URL to run
# anywhere else. Set before weave.init() runs — the SDK reads it at init.
# ---------------------------------------------------------------------------

CAP1_BASE_URL = "https://wandb.cloud.capitalone.com"

WANDB_BASE_URL = os.environ.setdefault("WANDB_BASE_URL", CAP1_BASE_URL).rstrip("/")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "api-evals-101")


def _check(name: str, value: str | None, example: str) -> None:
    if not value or "<" in value:
        raise SystemExit(
            f"{name} is not set.\n"
            f"    export {name}={example}\n"
            "    (or copy .env.example to .env, fill it in, and `source .env`)"
        )


def project() -> str:
    _check("WANDB_ENTITY", WANDB_ENTITY, "<your-team>")
    return f"{WANDB_ENTITY}/{WANDB_PROJECT}"


def ui_url() -> str:
    """The Weave UI for this project (the public cloud UI lives at wandb.ai)."""
    host = WANDB_BASE_URL.replace("://api.wandb.ai", "://wandb.ai")
    return f"{host}/{WANDB_ENTITY}/{WANDB_PROJECT}/weave"


# ---------------------------------------------------------------------------
# The three candidate models. Same bot, same prompt, same retriever:
# the model is the ONLY thing that changes between eval runs.
#
# These are gateway model IDs, and FM Hosting renames them (the provider
# prefix, e.g. `openai/`, was added in a rename — bare `gpt-oss-20b` no longer
# resolves). Run `python 00_check_setup.py`: it pings each one and, on a
# failure, prints the IDs the gateway is actually serving. Override with
# MODEL_A / MODEL_B / MODEL_C.
#
# `provider` is not a vendor choice here — every call goes through the same
# gateway. It is the string Weave records as gen_ai.provider.name.
# ---------------------------------------------------------------------------

GATEWAY = "capitalone-gateway"

CANDIDATES: dict[str, dict] = {
    "gpt-oss-120b": {
        "provider": GATEWAY,
        "model": os.environ.get("MODEL_A", "openai/gpt-oss-120b"),
    },
    "gpt-oss-20b": {
        "provider": GATEWAY,
        "model": os.environ.get("MODEL_B", "openai/gpt-oss-20b"),
    },
    "llama-4-scout": {
        "provider": GATEWAY,
        "model": os.environ.get("MODEL_C", "meta/llama-4-scout"),
    },
}

# ---------------------------------------------------------------------------
# The judge. ONE fixed model grades every candidate, and it should NOT be one
# of the candidates: LLM judges tend to prefer their own outputs
# (self-preference bias), so a candidate grading itself skews the comparison.
#
# The gateway serves a short list, so the default judge below IS one of the
# candidates. That is a known compromise, not an endorsement — point
# JUDGE_MODEL at a served model that isn't in CANDIDATES as soon as there is
# one. `warn_about_mocks()` prints a reminder when they overlap.
# ---------------------------------------------------------------------------

JUDGE = {
    "provider": GATEWAY,
    "model": os.environ.get("JUDGE_MODEL", "openai/gpt-oss-120b"),
}

# Optional: set TEMPERATURE to force a value. Left unset by default because
# some reasoning models reject any temperature other than their default.
TEMPERATURE = os.environ.get("TEMPERATURE")

# How many times to run each row. >1 shows you run-to-run noise.
TRIALS = int(os.environ.get("TRIALS", "1"))

# BACKEND=mock runs the whole pipeline with no gateway calls (plumbing check
# only; see mock_llm.py). Otherwise calls are real whenever the gateway client
# is importable, and fall back to the mock when it isn't — so running this on a
# laptop outside the sandbox never blocks the session.
BACKEND = os.environ.get("BACKEND", "real").lower()


def gateway_available() -> bool:
    """True if Capital One's inference client is importable in this env."""
    try:
        import c1.aiml.genai.inference  # noqa: F401

        return True
    except Exception:  # noqa: BLE001 — any import failure means "not here"
        return False


def uses_mock(provider: str) -> bool:
    """True if calls go to the mock (BACKEND=mock, or no gateway in this env)."""
    return BACKEND == "mock" or not gateway_available()


def mocked_candidates(names: list[str] | None = None) -> list[str]:
    return [n for n in (names or list(CANDIDATES))
            if uses_mock(CANDIDATES[n]["provider"])]


def warn_about_mocks(names: list[str], judge: bool = True) -> None:
    """Say loudly which parts of this run are scripted, not real models."""
    if BACKEND != "mock" and not gateway_available():
        print("NOTE: c1.aiml.genai.inference is not importable here -> every model "
              "runs on the scripted mock (labelled '-mock'). Mock results say "
              "nothing about any model. Run inside the Capital One sandbox.")
    if judge and JUDGE["model"] in {c["model"] for c in CANDIDATES.values()}:
        print(f"NOTE: the judge ({JUDGE['model']}) is also a candidate, so it grades "
              "its own answers. Set JUDGE_MODEL to a served model that isn't in "
              "CANDIDATES when one is available.")
    if judge and uses_mock(JUDGE["provider"]):
        print("NOTE: the LLM judge is mocked and always says yes, so handles_scope "
              "and grounded_in_articles are not real checks this run.")


def eval_display_name(scope: str) -> str:
    """<YYYYMMDDTHHMM>__<scope> so the Evaluations tab sorts cleanly."""
    return f"{_dt.datetime.now().strftime('%Y%m%dT%H%M')}__{scope}"

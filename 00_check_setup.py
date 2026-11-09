"""00 — Pre-flight check. Run this before the workshop starts.

Checks that:
  - WANDB_ENTITY is set and the W&B host looks right (values are never printed)
  - `c1_aiml_aem` and the gateway client import in this environment
  - each candidate model and the judge answer a one-line ping
It doesn't log anything to Weave.

If a model ID is rejected, this script asks the gateway which IDs it is
serving and prints them. Set MODEL_A / MODEL_B / MODEL_C (or JUDGE_MODEL) to
one of those and rerun — FM Hosting renames models, so the defaults in
config.py go stale.
"""

from __future__ import annotations

import os

import config
from llm import chat

# `chat` is a @weave.op. These pings run before weave.init(), so call the
# undecorated function and keep Weave's "traces will not be logged" warning
# out of a pre-flight that deliberately logs nothing.
ping = getattr(chat, "resolve_fn", chat)


def served_model_ids() -> list[str]:
    """Ask the gateway what it serves. Best-effort: not every deployment of
    the client exposes `models.list()`."""
    from c1.aiml.genai.inference import Client as C1Client

    return sorted(m.id for m in C1Client().models.list())


def main() -> None:
    print("W&B host:     ", config.WANDB_BASE_URL)
    print("Weave project:", f"{config.WANDB_ENTITY}/{config.WANDB_PROJECT}"
          if config.WANDB_ENTITY else "WANDB_ENTITY NOT SET (your team name)")

    try:
        import c1_aiml_aem  # noqa: F401
        print("  c1_aiml_aem       importable (Weave comes from the wrapper)")
    except ImportError:
        print("  c1_aiml_aem       MISSING -> falling back to the public `weave` "
              "package. Fine on a laptop, not what you run inside C1.")

    status = "set" if os.environ.get("WANDB_API_KEY") else "MISSING"
    if status == "MISSING" and os.path.exists(os.path.expanduser("~/.netrc")):
        status = "not set (using `wandb login` from ~/.netrc)"
    print(f"  WANDB_API_KEY     {status}")

    gw = config.gateway_available()
    print(f"  inference gateway {'importable' if gw else 'MISSING (c1.aiml.genai.inference)'}")

    if config.BACKEND == "mock":
        print("BACKEND=mock: skipping live pings.")
        return
    if not gw:
        print("\nNo gateway in this environment, so every model would run on the "
              "scripted mock. Run inside the Capital One sandbox for real numbers.")
        return

    targets = [(n, c["provider"], c["model"]) for n, c in config.CANDIDATES.items()]
    targets.append(("judge", config.JUDGE["provider"], config.JUDGE["model"]))
    ok = True
    listed = False
    for label, provider, model in targets:
        try:
            r = ping(provider, model, "Reply with the single word: ready", "ping")
            print(f"  OK   {label:<14} {model:<28} -> {r.text.strip()[:30]!r}")
        except Exception as e:  # noqa: BLE001 — show any failure plainly
            ok = False
            print(f"  FAIL {label:<14} {model:<28} -> {type(e).__name__}: {str(e)[:120]}")
            if not listed:
                listed = True
                try:
                    print(f"       gateway is serving: {', '.join(served_model_ids()[:25])}")
                except Exception:  # noqa: BLE001 — models.list() is best-effort
                    print("       (couldn't list served models; ask the FM Hosting "
                          "team which IDs are live)")
    if config.WANDB_ENTITY:
        print(f"\nWeave UI: {config.ui_url()}")
    print("\nAll good." if ok else "\nFix the FAIL lines (usually a model ID) before running 01/02.")


if __name__ == "__main__":
    main()

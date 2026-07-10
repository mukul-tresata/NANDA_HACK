"""Entrypoint: `python -m service.serve` (run from the repo root).

Serves the Conductor-Delta agent over HTTP. Host/port come from env so the same
command works locally and behind a tunnel:

    HOST=0.0.0.0 PORT=6001 python -m service.serve

Then expose it publicly (see service/README.md) -- e.g. a Cloudflare Tunnel --
to get the URL you put in SKILL.md.
"""

import os

import uvicorn


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "6001"))
    # Single worker on purpose: one shared Orchestrator, run() serialized behind
    # a lock. Multiple workers would each hold their own model state and race.
    uvicorn.run("service.app:app", host=host, port=port, workers=1, log_level="info")


if __name__ == "__main__":
    main()

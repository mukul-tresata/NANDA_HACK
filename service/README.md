# CEO-Delta HTTP service — run & deploy

This is the NANDA-facing HTTP layer around the CEO-Delta planner. The core
`ceo_delta` package stays stdlib-only; only this `service/` package needs
fastapi/uvicorn.

## 1. Install service deps

```bash
pip install -r service/requirements.txt
```

## 2. Run locally (offline test mode)

Run from the **repo root** so `ceo_delta` and `service` both import:

```bash
# grounding off by default -> pure local test, no outward calls beyond the LLM
python -m service.serve
```

Defaults: `HOST=0.0.0.0`, `PORT=6001`. Override via env. The service points at
the vLLM backend from `ceo_delta/config.py`; override with
`CEO_DELTA_LLM_BASE_URL` if the model host differs.

Smoke test:

```bash
curl -s localhost:6001/health
curl -s -X POST localhost:6001/plan -H 'Content-Type: application/json' \
  -d '{"task":"Explain how a modern C compiler turns C source into an executable."}' | jq .
```

## 3. Expose it publicly (for NANDA / the hackathon)

The model is a heavy self-hosted vLLM, so run the service **on the model host**
and tunnel to it rather than migrating the model to a cloud PaaS. A Cloudflare
Tunnel gives a public HTTPS URL in one command, no domain or certs to manage:

```bash
# one-time: install cloudflared (https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)
# then, with the service already running on :6001 --
cloudflared tunnel --url http://localhost:6001
```

`cloudflared` prints a `https://<random>.trycloudflare.com` URL. That is your
public base URL — put it in `SKILL.md` (replace `https://YOUR-PUBLIC-URL`).

> Any public URL is accepted by the hackathon (Railway/Render/Fly/tunnel/etc.).
> The tunnel is chosen here only because the model can't be lifted off its host.

## 4. Mode toggle (local ↔ NANDA)

Everything is env-driven so the same service runs in both modes:

| env var                       | values                       | effect                                           |
|-------------------------------|------------------------------|--------------------------------------------------|
| `CEO_DELTA_GROUNDING_BACKEND` | `none` (default) / `web_search` / `nanda` | outward grounding/discovery mode          |
| `CEO_DELTA_WORKDIR`           | path (default `.ceo_delta`)  | persistent handbook/EF state dir                 |
| `CEO_DELTA_LLM_BASE_URL`      | URL                          | override the vLLM/OpenAI-compatible base URL      |
| `HOST` / `PORT`               | (default `0.0.0.0` / `6001`) | bind address                                     |

- **Local testing:** leave `CEO_DELTA_GROUNDING_BACKEND=none`. No NANDA needed.
- **NANDA mode:** set `CEO_DELTA_GROUNDING_BACKEND=nanda` once a NANDA directory
  endpoint is live (the `nanda` grounding branch is a stubbed seam in
  `ceo_delta/grounding.py` — wire it there).

## 5. Register with NANDA

1. Get the public URL from step 3 and paste it into `SKILL.md`.
2. Sign in at <https://index.projectnanda.org>, publish your Agent Facts, claim a handle.
3. Submit the skill via the form on the NANDA Town skills page.

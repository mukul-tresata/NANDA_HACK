"""HTTP service layer that exposes Conductor-Delta to the outside world.

This package is the ONLY part of the system that depends on third-party web
libraries (fastapi/uvicorn). The core `ceo_delta` package stays stdlib-only by
design -- the service imports it, never the other way round. If you only want to
run/test the model, you never touch this package; use `cli.py`.
"""

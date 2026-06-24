"""CEO-Delta Architecture — a self-improving multi-agent planning/execution loop.

Public surface:
    from ceo_delta import Orchestrator, Config
"""
from .config import Config, DEFAULT
from .orchestrator import Orchestrator, RunResult

__all__ = ["Orchestrator", "RunResult", "Config", "DEFAULT"]
__version__ = "0.1.0"

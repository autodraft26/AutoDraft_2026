"""Autodraft: thin wrapper API around the AutoDraft speculative decoding runtime.

This module is intentionally import-light. It exposes only ``Autodraft`` and
must not pull in ``torch``, ``transformers``, or any other heavy dependency at
import time. Heavy imports happen lazily inside ``autodraft.local_runner``.
"""

from .engine import Autodraft
from .target_server import serve_target

__all__ = ["Autodraft", "serve_target"]

__version__ = "0.1.0"

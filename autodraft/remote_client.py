"""Newline-delimited JSON socket client for the AutoDraft target server.

The AutoDraft target server (``evaluation/eval_autodraft_target.py``) speaks
newline-delimited JSON over a raw TCP socket. The helpers in
``opt_classic/utils.py`` (``send_json`` / ``recv_json``) implement that wire
format, but they live outside the ``autodraft`` wheel, so we re-implement the
exact same protocol here to keep this module dependency-free.
"""

from __future__ import annotations

import json
import socket
from typing import Any, Dict, Optional

from .errors import RemoteTargetConnectionError


def _send_json(sock: socket.socket, payload: Dict[str, Any]) -> None:
    data = (json.dumps(payload) + "\n").encode("utf-8")
    sock.sendall(data)


def _recv_json(sock: socket.socket) -> Dict[str, Any]:
    buffer = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("Socket closed by peer")
        buffer += chunk
        if b"\n" in buffer:
            line, _ = buffer.split(b"\n", 1)
            return json.loads(line.decode("utf-8"))


def request_remote_target(
    host: str,
    port: int,
    payload: Dict[str, Any],
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Send ``payload`` to a AutoDraft target server and return the first reply.

    Wraps any low-level socket / connection failure in
    :class:`RemoteTargetConnectionError` so callers don't have to catch
    multiple exception types. The wire format is newline-delimited JSON, which
    is what ``evaluation/eval_autodraft_target.py`` expects.
    """

    if not host:
        raise RemoteTargetConnectionError("target_host is required")
    if not isinstance(port, int):
        raise RemoteTargetConnectionError("target_port must be int")

    sock: Optional[socket.socket] = None
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.settimeout(timeout)
        _send_json(sock, payload)
        return _recv_json(sock)
    except socket.timeout as exc:
        raise RemoteTargetConnectionError(
            f"timed out talking to target server at {host}:{port} "
            f"(timeout={timeout}s)"
        ) from exc
    except (ConnectionRefusedError, ConnectionError, OSError) as exc:
        raise RemoteTargetConnectionError(
            f"failed to reach target server at {host}:{port}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RemoteTargetConnectionError(
            f"target server at {host}:{port} returned non-JSON response: {exc}"
        ) from exc
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

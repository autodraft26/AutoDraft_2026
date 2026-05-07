"""Custom exception hierarchy for autodraft."""


class AutodraftError(Exception):
    """Base class for all autodraft errors."""


class InvalidAutodraftConfigError(AutodraftError):
    """Raised when ``Autodraft`` is constructed with invalid arguments."""


class RemoteTargetConnectionError(AutodraftError):
    """Raised when the client cannot reach or talk to the remote target server."""


class LocalRuntimeUnavailableError(AutodraftError):
    """Raised when the in-process local runner is requested but its heavy
    dependencies (torch, transformers, the source-checkout AutoDraft modules)
    are not importable. The standalone PyPI wheel ships only the wrapper and
    the wire protocol, so local execution requires a source checkout plus the
    optional ``[local]`` extra."""

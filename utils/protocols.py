"""Shared benchmark protocol names and capability errors."""

FIXED_PROTOCOL = "fixed"
NATIVE_PROTOCOL = "native"
PROTOCOLS = (FIXED_PROTOCOL, NATIVE_PROTOCOL)


class ProtocolUnavailableError(RuntimeError):
    """Raised when a framework cannot represent a dataset in a protocol."""


class ProtocolSkippedError(RuntimeError):
    """Raised when an optional external dependency was deliberately not invoked."""


def validate_protocol(protocol):
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown benchmark protocol {protocol!r}; expected one of {PROTOCOLS}")
    return protocol


def benchmark_exit_code(rows, allow_empty=False):
    """Return failure for errors or for a run that produced no measurements."""

    if any(row.get("status") == "error" for row in rows):
        return 1
    if not allow_empty and not any(row.get("status") == "ok" for row in rows):
        return 2
    return 0

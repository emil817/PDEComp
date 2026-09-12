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

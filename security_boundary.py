"""Local-development exposure checks, not authentication or a proxy trust policy.

A tunnel can rewrite Host/Origin and forward from loopback. These checks cannot
detect that arrangement: do not tunnel or remotely proxy the unauthenticated UI.
"""
from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit
from uuid import uuid4


def is_loopback_literal(value: object) -> bool:
    """Accept only numeric loopback addresses, without DNS resolution or coercion."""
    if not isinstance(value, str) or not value or "%" in value:
        return False
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def valid_api_key(value: object) -> bool:
    """Keys must be printable ASCII HTTP header tokens; never normalise secrets."""
    return isinstance(value, str) and 1 <= len(value) <= 4096 and all(
        33 <= ord(char) <= 126 for char in value
    )


def _local_authority(value: object, *, origin: bool = False) -> bool:
    if not isinstance(value, str) or not value or any(c.isspace() for c in value):
        return False
    try:
        parsed = urlsplit(value if origin else "//" + value)
        host = parsed.hostname
        port = parsed.port  # also validate a malformed/out-of-range port
        return bool(
            host and (host == "localhost" or is_loopback_literal(host))
            and parsed.username is None and parsed.password is None
            and not parsed.path and not parsed.query and not parsed.fragment
            and (not origin or parsed.scheme in {"http", "https"})
            and (port is None or 1 <= port <= 65535)
        )
    except ValueError:
        return False


def ui_boundary_problems(get_option, headers) -> list[str]:
    """Check effective (including CLI/env overrides) settings before UI work.

    Host/Origin are rejection signals only, not trusted identity. Unknown session
    context fails closed. This guards app execution, not Streamlit's HTTP server.
    """
    problems = []
    if not is_loopback_literal(get_option("server.address")):
        problems.append("The unauthenticated UI requires a numeric loopback server.address.")
    if get_option("server.enableCORS") is not True:
        problems.append("The local UI requires server.enableCORS=true.")
    if get_option("server.enableXsrfProtection") is not True:
        problems.append("The local UI requires server.enableXsrfProtection=true.")
    browser_address = get_option("browser.serverAddress")
    if browser_address != "localhost" and not is_loopback_literal(browser_address):
        problems.append("The local UI requires a loopback browser.serverAddress.")
    lowered = {str(key).lower(): value for key, value in headers.items()}
    if hasattr(headers, "get_all") and any(
        len(headers.get_all(name)) > 1 for name in ("host", "origin")
    ):
        problems.append("The local UI rejects ambiguous Host/Origin headers.")
    if not _local_authority(lowered.get("host")):
        problems.append("The local UI requires a local Host header.")
    if "origin" in lowered and not _local_authority(lowered["origin"], origin=True):
        problems.append("The local UI rejects non-local or malformed Origin headers.")
    if any(key == "forwarded" or key.startswith("x-forwarded-") for key in lowered):
        problems.append("Forwarded/proxied UI sessions are unsupported.")
    return problems


def session_resource(state, key: str, factory):
    """Store private mutable resources in session state, never Streamlit caches."""
    if key not in state:
        state[key] = factory()
    return state[key]


def session_cache(state, architecture: str, factory):
    caches = session_resource(state, "_private_answer_caches", dict)
    scope = session_resource(state, "_answer_cache_scope", lambda: uuid4().hex)
    if architecture not in caches:
        caches[architecture] = factory(f"ui-session:{scope}:{architecture}")
    return caches[architecture]

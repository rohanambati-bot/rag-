"""Static deployment contract checks, not a substitute for a Linux image build."""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_container_default_is_authenticated_api_not_remote_ui():
    text = (ROOT / "Dockerfile").read_text()
    command = json.loads(next(line[4:] for line in text.splitlines() if line.startswith("CMD ")))
    assert command[:2] == ["uvicorn", "api:app"]
    assert "--no-proxy-headers" in command
    assert "/healthz" in text and "libgomp1" in text
    assert "USER appuser" in text


def test_compose_does_not_publish_unauthenticated_ui_or_allow_auth_optout():
    services = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]
    assert set(services) == {"api", "redis"}
    api = services["api"]
    assert "ALLOW_UNAUTHENTICATED=false" in api["environment"]
    assert "HOST=0.0.0.0" in api["environment"]
    assert "--no-proxy-headers" in api["command"]
    for service in services.values():
        assert all(port.startswith("127.0.0.1:") for port in service.get("ports", []))

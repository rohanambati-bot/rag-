"""Default UI exposure is local-only; this is not a UI authentication system."""

from pathlib import Path
import tomllib

import streamlit.config as streamlit_config
import pytest

from security_boundary import ui_boundary_problems


ROOT = Path(__file__).resolve().parents[1]


def test_project_streamlit_configuration_is_loopback_only():
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
    assert config["server"]["address"] == "127.0.0.1"
    assert config["server"]["enableCORS"] is True
    assert config["server"]["enableXsrfProtection"] is True
    assert config["server"]["maxUploadSize"] == 20


def test_streamlit_loads_project_loopback_setting(monkeypatch):
    monkeypatch.chdir(ROOT)
    # A real Streamlit config parse proves these are recognised option names;
    # no model, browser, dataset, provider call or listening server is started.
    options = streamlit_config.get_config_options(force_reparse=True)
    assert options["server.address"].value == "127.0.0.1"
    assert options["server.enableXsrfProtection"].value is True


def _settings(**overrides):
    return {"server.address": "127.0.0.1", "server.enableCORS": True,
            "server.enableXsrfProtection": True, "browser.serverAddress": "localhost",
            **overrides}


@pytest.mark.parametrize("host", ["", "0.0.0.0", "::", "localhost", "127.0.0.1:8501",
                                  "127.0.0.1.evil", "127.0.0.1\n", None])
def test_ui_rejects_remote_and_malformed_bind_settings(host):
    problems = ui_boundary_problems(_settings(**{"server.address": host}).get,
                                    {"Host": "localhost:8501"})
    assert any("server.address" in problem for problem in problems)


@pytest.mark.parametrize("headers", [
    {}, {"Host": "app.example.com"}, {"Host": "localhost:bad"},
    {"Host": "localhost.evil"}, {"Host": "localhost:8501", "Origin": "null"},
    {"Host": "localhost:8501", "Origin": "https://app.example.com"},
    {"Host": "localhost:8501", "Forwarded": "for=127.0.0.1"},
    {"Host": "localhost:8501", "X-Forwarded-Host": "localhost"},
])
def test_ui_rejects_unknown_remote_or_explicitly_forwarded_context(headers):
    assert ui_boundary_problems(_settings().get, headers)


def test_ui_rejects_ambiguous_session_headers():
    from streamlit.runtime.context import StreamlitHeaders
    headers = StreamlitHeaders([("Host", "localhost:8501"), ("Host", "evil.example")])
    assert ui_boundary_problems(_settings().get, headers)


@pytest.mark.parametrize("address,host", [("127.0.0.1", "localhost:8501"),
                                          ("::1", "[::1]:8501")])
def test_direct_local_settings_work_without_claiming_authentication(address, host):
    assert ui_boundary_problems(_settings(**{"server.address": address}).get,
                                {"Host": host, "Origin": "http://" + host}) == []


@pytest.mark.parametrize("key,value", [("server.enableCORS", False),
                                       ("server.enableXsrfProtection", False),
                                       ("browser.serverAddress", "app.example.com")])
def test_ui_rejects_weakened_settings(key, value):
    assert ui_boundary_problems(_settings(**{key: value}).get, {"Host": "localhost:8501"})


@pytest.mark.parametrize("source", ["env", "cli"])
def test_actual_streamlit_override_is_checked_and_not_silently_allowed(monkeypatch, source):
    from click.testing import CliRunner
    from streamlit.web import bootstrap, cli
    captured = {}

    def capture_start(file, args, flag_options):
        captured.update(flag_options)

    monkeypatch.setattr(cli, "_main_run", capture_start)
    if source == "env":
        monkeypatch.setenv("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")
    try:
        args = ["run", str(ROOT / "app.py")]
        if source == "cli":
            args += ["--server.address", "0.0.0.0"]
        result = CliRunner().invoke(cli.main, args)
        assert result.exit_code == 0, result.output
        bootstrap.load_config_options(flag_options=captured)
        assert streamlit_config.get_option("server.address") == "0.0.0.0"
        assert ui_boundary_problems(streamlit_config.get_option, {"Host": "localhost:8501"})
    finally:
        monkeypatch.delenv("STREAMLIT_SERVER_ADDRESS", raising=False)
        streamlit_config.get_config_options(force_reparse=True, options_from_flags={})


def test_actual_app_stops_before_resource_work_for_unsafe_effective_settings(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import serving
    monkeypatch.setattr(serving.CONTEXT, "embedding", lambda: pytest.fail("must not load models"))
    monkeypatch.setattr(serving.CONTEXT, "architectures", lambda: pytest.fail("must not build RAG"))
    try:
        streamlit_config.set_option("server.address", "0.0.0.0")
        app = AppTest.from_file(str(ROOT / "app.py")).run()
        assert not app.exception
        assert any("UI access refused" in error.value for error in app.error)
        assert not app.sidebar.radio
    finally:
        streamlit_config.get_config_options(force_reparse=True, options_from_flags={})

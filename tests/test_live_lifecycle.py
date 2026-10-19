"""Opt-in real processes; skipped explicitly in deterministic offline make test.

LIVE_LIFECYCLE=1 REDIS_SERVER=<binary> PYTHON_DOTENV_DISABLED=1 \
    .venv/bin/python -m pytest -q tests/test_live_lifecycle.py
"""
import os
import shutil

import pytest

pytestmark = pytest.mark.skipif(os.getenv("LIVE_LIFECYCLE") != "1",
                                reason="opt-in real processes: set LIVE_LIFECYCLE=1 (see module command)")


def test_real_redis_separate_worker_lifecycle():
    binary = os.getenv("REDIS_SERVER") or shutil.which("redis-server")
    if not binary:
        pytest.skip("redis-server unavailable; set REDIS_SERVER to a local binary")
    from scripts.validate_live_lifecycle import validate_redis
    result = validate_redis(binary)
    assert result["status"] == "passed"
    assert result["distinct_worker_pids"] == result["worker_count"]


def test_real_chroma_concurrent_reader_publication_and_rollback():
    from scripts.validate_live_lifecycle import validate_index_readers
    assert validate_index_readers()["status"] == "passed"

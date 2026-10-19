"""Offline boundary tests for the advisory matrix; no scanner or model calls."""

import json

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement

from scripts.dependency_security_report import (
    advisory_groups,
    build_report,
    dependency_graph,
    dependency_paths,
    fixed_version_constraints,
    main,
    read_requirements,
    source_inventory,
    write_report,
)


def test_aliases_merge_transitively_but_preserve_package_exposure():
    dependencies = [
        {"name": "Foo_Bar", "version": "1", "vulns": [
            {"id": "PYSEC-1", "aliases": ["GHSA-a"], "fix_versions": ["2"]},
            {"id": "CVE-1", "aliases": ["GHSA-a"], "fix_versions": []},
        ]},
        {"name": "baz", "version": "3", "vulns": [
            {"id": "CVE-1", "aliases": ["other"], "fix_versions": ["4"]},
        ]},
    ]
    rows = advisory_groups(dependencies)
    assert len(rows) == 2
    assert {row["advisory"] for row in rows} == {"CVE-1"}
    assert all(row["identifiers"] == ["CVE-1", "GHSA-a", "PYSEC-1", "other"] for row in rows)
    foo = next(row for row in rows if row["package"] == "foo-bar")
    assert foo["scanner_record_count"] == 2
    assert foo["fix_versions_supplied"] == ["2"]  # Do not borrow baz's fixes.
    assert foo["severity"] == foo["affected_range"] == "unknown"


def test_graph_activates_transitive_extras_and_ignores_inactive_markers():
    roots = {"root": Requirement("root[net]==1")}
    installed = {
        "root": {"version": "1", "requires_dist": ["middle[http]>=2; extra == 'net'", "hidden; python_version < '2'"]},
        "middle": {"version": "2", "requires_dist": ["target>=3; extra == 'http'", "root"]},
        "target": {"version": "3", "requires_dist": []},
    }
    environment = default_environment()
    edges = dependency_graph(installed, roots, environment)
    assert {edge["child"] for edge in edges} == {"middle", "root", "target"}
    assert all(edge["version_satisfies"] for edge in edges)
    assert dependency_paths(roots, edges, "target", environment) == [["root", "middle", "target"]]
    assert dependency_paths(roots, edges, "hidden", environment) == []


def test_audit_lock_environment_drift_is_visible_and_never_suppressed():
    roots = {"foo": Requirement("foo==1")}
    installed = {"foo": {"version": "2", "requires_dist": []}}
    audit = {"dependencies": [{"name": "foo", "version": "1", "vulns": [{"id": "CVE-1", "fix_versions": []}]}]}
    report = build_report(audit, roots, roots, installed, {"imports": []}, default_environment())
    assert {issue["kind"] for issue in report["consistency_issues"]} == {
        "lock_version_mismatch", "direct_requirement_mismatch", "audit_version_mismatch",
    }
    assert report["summary"]["open_rows"] == 1
    row = report["advisories"][0]
    assert row["disposition"]["status"] == "open"
    assert row["reachability"]["status"] == "not exploit-verified"
    assert row["fix_versions_supplied"] == []


def test_source_inventory_covers_lazy_imports_without_reading_private_files(tmp_path):
    (tmp_path / "app.py").write_text("def load():\n    from sentence_transformers import CrossEncoder\n")
    (tmp_path / ".env").write_bytes(b"\xff private, not Python")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_fake.py").write_text("invalid python !")
    result = source_inventory(tmp_path)
    assert list(result["files"]) == ["app.py"]
    assert result["imports"] == [{"path": "app.py", "line": 2, "module": "sentence_transformers", "symbols": ["CrossEncoder"]}]


def test_requirement_reader_keeps_url_hash_and_rejects_duplicate_roots(tmp_path):
    path = tmp_path / "requirements.txt"
    path.write_text("foo>=1 # explanation\nbar @ https://example.invalid/bar.whl#sha256=abc\n")
    requirements = read_requirements(path)
    assert requirements["bar"].url.endswith("#sha256=abc")
    path.write_text("Foo==1\nfoo==2\n")
    with pytest.raises(ValueError, match="duplicate requirement"):
        read_requirements(path)


def test_evidence_replay_is_byte_reproducible_and_refuses_overwrite(tmp_path, monkeypatch):
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"dependencies": [{"name": "foo", "version": "1", "vulns": [{"id": "CVE-1"}]}]}))
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("foo==1\n")
    monkeypatch.setattr("scripts.dependency_security_report.collect_metadata", lambda _: {"foo": {"version": "1", "requires_dist": []}})
    first, second = tmp_path / "first", tmp_path / "second"
    args = ["--audit", str(audit), "--requirements", str(requirements), "--lock", str(requirements), "--project-root", str(tmp_path)]
    assert main([*args, "--output-dir", str(first)]) == 0
    monkeypatch.setattr("scripts.dependency_security_report.collect_metadata", lambda _: pytest.fail("replay must not read environment"))
    assert main([*args, "--output-dir", str(second), "--evidence", str(first / "evidence.json")]) == 0
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    with pytest.raises(FileExistsError):
        write_report(first, {}, {})
    with pytest.raises(SystemExit):
        main([*args, "--output-dir", str(first)])
    requirements.write_text("foo==2\n")
    with pytest.raises(SystemExit):
        main([*args, "--output-dir", str(tmp_path / "third"), "--evidence", str(first / "evidence.json")])


def test_scanner_skips_remain_coverage_gaps():
    audit = {"dependencies": [{"name": "model", "skip_reason": "direct URL"}], "fixes": []}
    report = build_report(audit, {}, {}, {}, {"imports": []}, default_environment())
    assert report["coverage_gaps"] == audit["dependencies"]
    assert report["summary"]["clean_audit"] is None


def test_supplied_fix_blockers_use_declared_constraints_not_version_guesses():
    roots = {"torch": Requirement("torch==2.5.1")}
    edges = [{"parent": "torchvision", "child": "torch", "requires_dist": "torch==2.5.1"}]
    review = fixed_version_constraints("torch", ["2.6.0"], roots, edges)
    assert review[0]["status"] == "blocked by retained constraints"
    assert {blocker["source"] for blocker in review[0]["blockers"]} == {"direct", "torchvision"}
    assert fixed_version_constraints("foo", ["2"], {}, [])[0]["status"].endswith("untested")


def test_review_notes_are_not_reused_for_a_different_installed_version():
    roots = {"transformers": Requirement("transformers==6")}
    installed = {"transformers": {"version": "6", "requires_dist": []}}
    audit = {"dependencies": [{"name": "transformers", "version": "6", "vulns": [{"id": "CVE-future"}]}]}
    report = build_report(audit, roots, roots, installed, {"imports": []}, default_environment())
    assert report["advisories"][0]["reachability"]["package_level_review"].startswith("Unknown;")


def test_indirect_model_use_is_attributed_to_the_actual_imported_integration():
    roots = {"sentence-transformers": Requirement("sentence-transformers==3.3.1")}
    lock = {**roots, "transformers": Requirement("transformers==4.46.3")}
    installed = {
        "sentence-transformers": {"version": "3.3.1", "requires_dist": ["transformers<5"]},
        "transformers": {"version": "4.46.3", "requires_dist": []},
    }
    sources = {"imports": [{"module": "sentence_transformers", "path": "app.py", "line": 3, "symbols": ["CrossEncoder"]}]}
    audit = {"dependencies": [{"name": "transformers", "version": "4.46.3", "vulns": [{"id": "CVE-1"}]}]}
    row = build_report(audit, roots, lock, installed, sources, default_environment())["advisories"][0]
    assert row["source_imports"] == []
    assert row["integration_source_imports"][0]["distribution"] == "sentence-transformers"
    assert row["integration_source_imports"][0]["path"] == "app.py"

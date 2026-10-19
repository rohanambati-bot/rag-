"""Offline, create-only pip-audit triage with reproducible metadata/source evidence.

Run from the project environment; this imports no application or model packages.
It does not query advisory services, install packages, or change requirements.
Severity and affected ranges are deliberately unknown: pip-audit fix versions
are not complete affected ranges, and descriptions are not verified CVSS data.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata as metadata
import json
from collections import defaultdict, deque
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version


IMPORTS = {
    "chromadb": "chromadb",
    "datasets": "datasets",
    "fsspec": "fsspec",
    "langchain": "langchain",
    "langchain-core": "langchain_core",
    "langchain-text-splitters": "langchain_text_splitters",
    "sentence-transformers": "sentence_transformers",
    "torch": "torch",
    "transformers": "transformers",
    "torchvision": "torchvision",
    "langchain-community": "langchain_community",
    "langchain-chroma": "langchain_chroma",
    "langchain-huggingface": "langchain_huggingface",
}

# Package-level review notes, never per-advisory exploitability assertions.
TRIAGE = {
    "chromadb": (
        "Embedded vector-store use is present. The follow-up identifies server/model-repository "
        "endpoints; source imports do not prove those endpoints exposed or unreachable.",
        "No fix supplied in this receipt; obtain upstream guidance for embedded versus server "
        "use. Do not substitute a network Chroma server without reviewing its boundary.",
    ),
    "datasets": (
        "Dataset loading is used. Folder-builder exploitation has not been tested; trusted "
        "dataset names do not establish safety of arbitrary metadata.",
        "Migrate datasets and fsspec together. Installed datasets 3.1.0 requires fsspec<=2024.9.0; "
        "the audit supplies datasets 5.0.1 and fsspec 2026.6.0, not a tested compatible pair. "
        "The earlier 5.1.0 attempt conflicts with the retained Hugging Face Hub <1 stack.",
    ),
    "fsspec": (
        "Installed through datasets, Hugging Face Hub and Torch metadata paths. No direct fsspec import is not proof "
        "that reference-filesystem template handling cannot be invoked transitively.",
        "datasets 3.1.0 caps fsspec<=2024.9.0, excluding the supplied fix 2026.6.0. "
        "Migrate the loader stack in an isolated environment and test real project behavior.",
    ),
    "langchain": (
        "No direct package import alone does not justify removal: langchain-community declares "
        "langchain as a runtime dependency. Named vulnerable helper calls remain unproven.",
        "Supplied fix 1.3.9 is outside direct <0.4. Removing the direct requirement leaves "
        "the transitive dependency; a coordinated integration migration needs tests.",
    ),
    "langchain-core": (
        "Document/embedding interfaces are imported. Prompt-file loading/image URL token "
        "counting reachability is not established by this source inventory.",
        "Supplied 1.x fixes conflict with direct <1.0 and installed langchain-huggingface "
        "0.2.0's langchain-core<1.0.0 constraint. Migrate the integration together.",
    ),
    "langchain-text-splitters": (
        "RecursiveCharacterTextSplitter is imported. This does not prove the advisory's "
        "HTML/URL loader is exposed or unreachable through other code.",
        "Supplied fix 1.1.2 is outside direct <0.4. Verify a compatible LangChain integration "
        "set and ingestion behavior before changing the constraints.",
    ),
    "sentence-transformers": (
        "SentenceTransformer/CrossEncoder model loading is present. Model-import/trust paths "
        "are security relevant; use trusted pinned model artifacts while migration is open.",
        "Supplied fix 5.6.0 changes the retained 3.3.1 model stack. Test model loading and "
        "retrieval/reranking contracts in an isolated candidate environment.",
    ),
    "torch": (
        "Torch device/model stack is used. Checkpoint and native-operation advisories require "
        "individual feature review; absence of torch.load in project code is not an exemption.",
        "torchvision 0.20.1 requires torch==2.5.1. Supplied fixes span later versions and some "
        "are empty; no single patch-only clean solution follows from this receipt. "
        "Coordinate Torch/vision (or verify removal) and run native/model checks.",
    ),
    "transformers": (
        "Model loading is reachable through Sentence Transformers/Hugging Face embeddings; "
        "individual processor/conversion advisory paths remain unverified.",
        "A 4.x update could only partially remediate this receipt: supplied fixes include "
        "5.0.0, 5.3.0 and 5.10.0; sentence-transformers 3.3.1 requires transformers<5.0.0. "
        "A coupled model-stack migration and tokenizers update require isolated tests.",
    ),
}

REVIEWED_VERSIONS = {
    "chromadb": "1.5.9", "datasets": "3.1.0", "fsspec": "2024.9.0", "langchain": "0.3.30",
    "langchain-core": "0.3.86", "langchain-text-splitters": "0.3.11", "sentence-transformers": "3.3.1",
    "torch": "2.5.1", "transformers": "4.46.3",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_requirements(path: Path) -> dict[str, Requirement]:
    result = {}
    for line in path.read_text().splitlines():
        line = line.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        req = Requirement(line)  # Reject unsupported pip directives rather than silently skip.
        name = canonicalize_name(req.name)
        if name in result:
            raise ValueError(f"duplicate requirement: {name}")
        result[name] = req
    return result


def advisory_groups(dependencies: list[dict]) -> list[dict]:
    """Union transitively overlapping IDs/aliases, preserving package exposures."""
    records = []
    parents = {}

    def find(value):
        parents.setdefault(value, value)
        if parents[value] != value:
            parents[value] = find(parents[value])
        return parents[value]

    for dependency in dependencies:
        for vuln in dependency.get("vulns", []):
            ids = sorted({vuln["id"], *vuln.get("aliases", [])})
            if not all(isinstance(value, str) and value for value in ids):
                raise ValueError("advisory IDs must be nonempty strings")
            for value in ids:
                parents[find(value)] = find(ids[0])
            records.append((dependency, vuln, ids))
    grouped = defaultdict(list)
    for record in records:
        grouped[find(record[2][0])].append(record)
    result = []
    for group in grouped.values():
        ids = sorted({value for _, _, values in group for value in values})
        canonical = next((value for value in ids if value.startswith("CVE-")), ids[0])
        exposures = defaultdict(list)
        for dependency, vuln, _ in group:
            exposures[(canonicalize_name(dependency["name"]), dependency.get("version"))].append(vuln)
        for (package, version), vulns in sorted(exposures.items()):
            result.append({
                "advisory": canonical,
                "identifiers": ids,
                "package": package,
                "audited_version": version,
                "scanner_record_count": len(vulns),
                "scanner_ids": sorted({vuln["id"] for vuln in vulns}),
                "fix_versions_supplied": sorted({str(v) for vuln in vulns for v in vuln.get("fix_versions", [])}),
                "descriptions": sorted({vuln.get("description", "") for vuln in vulns}),
                "severity": "unknown",
                "affected_range": "unknown",
                "verification": "No independent severity/range verification; scanner matched audited version only.",
            })
    return sorted(result, key=lambda row: (row["package"], row["advisory"]))


def collect_metadata(lock: dict[str, Requirement]) -> dict:
    """Read installed distribution metadata, not imported package code or site paths."""
    result = {}
    for name in sorted(lock):
        try:
            dist = metadata.distribution(name)
            result[name] = {"version": dist.version, "requires_dist": sorted(dist.requires or [])}
        except metadata.PackageNotFoundError:
            result[name] = {"version": None, "requires_dist": [], "error": "not installed"}
    return result


def dependency_graph(installed: dict, roots: dict[str, Requirement], environment: dict) -> list[dict]:
    """Activate markers/extras to a fixed point; include raw evidence for every edge."""
    extras = {name: set(req.extras) for name, req in roots.items()}
    reached = {name for name, req in roots.items() if not req.marker or req.marker.evaluate(environment)}
    edges = {}
    changed = True
    while changed:
        changed = False
        for parent in sorted(reached):
            for raw in installed.get(parent, {}).get("requires_dist", []):
                req = Requirement(raw)
                contexts = ["", *sorted(extras.get(parent, set()))]
                if req.marker and not any(req.marker.evaluate({**environment, "extra": extra}) for extra in contexts):
                    continue
                child = canonicalize_name(req.name)
                version = installed.get(child, {}).get("version")
                edges[(parent, child, raw)] = {
                    "parent": parent, "child": child, "requires_dist": raw,
                    "installed_child_version": version,
                    "version_satisfies": bool(version and req.specifier.contains(version)),
                }
                if child not in reached:
                    reached.add(child)
                    changed = True
                previous = extras.setdefault(child, set()).copy()
                extras[child].update(req.extras)
                changed |= extras[child] != previous
    return [edges[key] for key in sorted(edges)]


def dependency_paths(roots: dict, edges: list[dict], target: str, environment: dict) -> list[list[str]]:
    """One deterministic shortest active metadata path per direct requirement."""
    graph = defaultdict(set)
    for edge in edges:
        graph[edge["parent"]].add(edge["child"])
    paths = []
    for root, req in sorted(roots.items()):
        if req.marker and not req.marker.evaluate(environment):
            continue
        queue = deque([[root]])
        visited = {root}
        while queue:
            path = queue.popleft()
            if path[-1] == target:
                paths.append(path)
                break
            for child in sorted(graph[path[-1]] - visited):
                visited.add(child)
                queue.append([*path, child])
    return paths


def fixed_version_constraints(package: str, versions: list[str], roots: dict, edges: list[dict]) -> list[dict]:
    """Compare supplied versions to existing constraints, without solving/installing.

    These are concrete blockers in the retained stack, not a prediction that a
    candidate is safe, available, or resolvable after changing those constraints.
    """
    constraints = [("direct", str(roots[package]))] if package in roots else []
    constraints.extend((edge["parent"], edge["requires_dist"]) for edge in edges if edge["child"] == package)
    result = []
    for version in versions:
        try:
            parsed = Version(version)
        except InvalidVersion:
            result.append({"supplied_version": version, "status": "invalid version; not evaluated", "blockers": []})
            continue
        blockers = [{"source": source, "requirement": raw} for source, raw in constraints
                    if not Requirement(raw).specifier.contains(parsed, prereleases=True)]
        result.append({"supplied_version": version, "status": "blocked by retained constraints" if blockers else "not blocked by inspected constraints; untested", "blockers": blockers})
    return result


def source_inventory(root: Path) -> dict:
    """Inventory runtime Python AST imports, including imports inside functions.

    Fixed source roots avoid scanning environments, datasets, private inputs,
    tests and historical outputs. Dynamic imports are explicitly not inferred.
    """
    files = list(root.glob("*.py"))
    for folder in ["module1", "module2", "module3", "module4", "module5", "phase3", "scripts"]:
        files.extend((root / folder).rglob("*.py"))
    result = {"files": {}, "imports": [], "scope": "root *.py, module1..5, phase3, scripts; excludes tests and dynamic imports"}
    for path in sorted(files):
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=relative)
        result["files"][relative] = sha256(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
                symbols = modules
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                modules = [node.module]
                symbols = [alias.name for alias in node.names]
            else:
                continue
            for module in modules:
                result["imports"].append({"path": relative, "line": node.lineno, "module": module, "symbols": sorted(symbols)})
    result["imports"].sort(key=lambda row: (row["path"], row["line"], row["module"]))
    return result


def build_report(audit: dict, roots: dict, lock: dict, installed: dict, sources: dict, environment: dict) -> dict:
    dependencies = audit["dependencies"]
    rows = advisory_groups(dependencies)
    edges = dependency_graph(installed, roots, environment)
    consistency = []
    for name, req in sorted(lock.items()):
        version = installed[name]["version"]
        if not version or (not req.url and not req.specifier.contains(version)):
            consistency.append({"package": name, "kind": "lock_version_mismatch", "installed": version, "requirement": str(req)})
    for name, req in sorted(roots.items()):
        if req.marker and not req.marker.evaluate(environment):
            continue
        version = installed.get(name, {}).get("version")
        if not version or not req.specifier.contains(version):
            consistency.append({"package": name, "kind": "direct_requirement_mismatch", "installed": version, "requirement": str(req)})
    for row in rows:
        name = row["package"]
        version = installed.get(name, {}).get("version")
        row["installed_version"] = version
        row["installed_matches_audit"] = version == row["audited_version"]
        if not row["installed_matches_audit"]:
            consistency.append({"package": name, "kind": "audit_version_mismatch", "installed": version, "audited": row["audited_version"]})
        row["direct_requirement"] = str(roots[name]) if name in roots else None
        row["dependency_paths"] = dependency_paths(roots, edges, name, environment)
        module = IMPORTS.get(name, name.replace("-", "_"))
        row["source_imports"] = [item for item in sources["imports"] if item["module"].split(".")[0] == module]
        # Attribute indirect use to actual imported integration packages along
        # recorded metadata paths; do not pretend it is a direct target import.
        integrations = sorted({part for path in row["dependency_paths"] for part in path[:-1] if part in IMPORTS})
        row["integration_source_imports"] = [
            {"distribution": integration, **item}
            for integration in integrations
            for item in sources["imports"]
            if item["module"].split(".")[0] == IMPORTS[integration]
        ]
        row["supplied_fix_constraint_review"] = fixed_version_constraints(name, row["fix_versions_supplied"], roots, edges)
        default_triage = (
            "Unknown; static imports and distribution dependency paths are not exploit testing.",
            "Open: review upstream advisory and verify candidate fixes in a separate environment.",
        )
        reachability, disposition = TRIAGE.get(name, default_triage) if version == REVIEWED_VERSIONS.get(name) else default_triage
        row["reachability"] = {"status": "not exploit-verified", "package_level_review": reachability}
        row["disposition"] = {"status": "open", "reason": disposition}
    return {
        "schema_version": 1,
        "summary": {
            "scanner_matches": sum(len(item.get("vulns", [])) for item in dependencies),
            "unique_advisories": len({row["advisory"] for row in rows}),
            "package_advisory_rows": len(rows),
            "vulnerable_packages": len({row["package"] for row in rows}),
            "open_rows": len(rows),
            "clean_audit": False if rows else None,
        },
        "limitations": [
            "Offline transformation of a supplied receipt, not a fresh advisory lookup or security sign-off.",
            "Severity and full affected ranges are unknown; fix_versions are scanner-supplied, not verified safe targets.",
            "Aliased records are unioned transitively; shared IDs across packages retain separate exposure rows.",
            "Dependency paths are shortest paths per direct root from active installed Requires-Dist metadata, not call graphs.",
            "Source imports are static evidence only; dynamic imports, model artifacts, native operations and exploitability are unverified.",
            "URL requirements are inventoried but artifact bytes are not independently verified; skipped scanner packages remain coverage gaps.",
            "Environment marker evaluation describes this interpreter/platform, not clean Linux resolution.",
            "No requirement changes or advisory suppressions are made by this report.",
        ],
        "coverage_gaps": [item for item in dependencies if "skip_reason" in item or "version" not in item],
        "url_requirements_not_artifact_verified": [str(req) for req in lock.values() if req.url],
        "scanner_fixes": audit.get("fixes", []),
        "consistency_issues": consistency,
        "dependency_edges": edges,
        "advisories": rows,
    }


def markdown_report(report: dict) -> str:
    summary = report["summary"]
    lines = ["# Dependency security v3 — offline advisory triage", "",
             f"{summary['scanner_matches']} scanner matches; {summary['unique_advisories']} alias-deduplicated advisories; "
             f"{summary['package_advisory_rows']} package/advisory rows in {summary['vulnerable_packages']} packages. "
             f"All {summary['open_rows']} rows remain open.", "",
             "## Boundaries", "", *[f"- {item}" for item in report["limitations"]], "",
             f"Installed/lock/audit consistency issues: {len(report['consistency_issues'])}.", "",
             "## Package decisions", ""]
    for name in sorted({row["package"] for row in report["advisories"]}):
        rows = [row for row in report["advisories"] if row["package"] == name]
        first = rows[0]
        lines.extend([f"### {name} {first['installed_version']} — {len(rows)} open rows", "",
                      first["reachability"]["package_level_review"], "", first["disposition"]["reason"], "",
                      "Metadata paths: " + "; ".join(" → ".join(path) for path in first["dependency_paths"]), "",
                      "Static direct import evidence: " + (", ".join(f"`{item['path']}:{item['line']}`" for item in first["source_imports"]) or "none in scanned scope"), "",
                      "Imported integrations on metadata paths: " + (", ".join(f"{item['distribution']} at `{item['path']}:{item['line']}`" for item in first["integration_source_imports"]) or "none in scanned scope"), ""])
    lines.extend(["## Advisory matrix", "", "Severity and affected range are **unknown for every row**. "
                  "Fix versions below are supplied by the scanner; an empty list means no supplied fix, not no possible fix.", "",
                  "| Package | Advisory (aliases in JSON/CSV) | Supplied fixes | Disposition |",
                  "|---|---|---|---|"])
    for row in report["advisories"]:
        lines.append(f"| {row['package']} | {row['advisory']} | {', '.join(row['fix_versions_supplied']) or 'none supplied'} | open |")
    return "\n".join(lines) + "\n"


def write_report(output: Path, report: dict, evidence: dict) -> None:
    # Directory creation is the create-only boundary: no historical files overwritten.
    output.mkdir(parents=False, exist_ok=False)
    for name, value in [("report.json", report), ("evidence.json", evidence)]:
        with (output / name).open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with (output / "report.md").open("x", encoding="utf-8") as handle:
        handle.write(markdown_report(report))
    fields = ["package", "audited_version", "installed_version", "advisory", "identifiers", "scanner_ids",
              "scanner_record_count", "severity", "affected_range", "fix_versions_supplied", "dependency_paths",
              "source_imports", "integration_source_imports", "supplied_fix_constraint_review", "reachability", "disposition"]
    with (output / "advisory_matrix.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in report["advisories"]:
            # JSON cells preserve lists/structured evidence without lossy delimiter conventions.
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, default=Path("requirements.txt"))
    parser.add_argument("--lock", type=Path, default=Path("requirements.lock"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, help="Replay previously captured evidence, without reading installed metadata/source")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("output directory already exists; choose a fresh versioned destination")
    audit = json.loads(args.audit.read_text())
    roots, lock = read_requirements(args.requirements), read_requirements(args.lock)
    inputs = {"audit_sha256": sha256(args.audit), "requirements_sha256": sha256(args.requirements), "lock_sha256": sha256(args.lock)}
    if args.evidence:
        evidence = json.loads(args.evidence.read_text())
        if evidence["inputs"] != inputs:
            parser.error("replay input hashes do not match captured evidence")
    else:
        evidence = {
            "inputs": inputs,
            "environment": default_environment(),
            "installed": collect_metadata(lock),
            "sources": source_inventory(args.project_root),
            "direct_requirements": {name: str(req) for name, req in sorted(roots.items())},
            "url_requirements_not_artifact_verified": [str(req) for req in lock.values() if req.url],
        }
    report = build_report(audit, roots, lock, evidence["installed"], evidence["sources"], evidence["environment"])
    report["input_hashes"] = inputs
    write_report(args.output_dir, report, evidence)
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""tests/test_no_cross_stack_imports.py — Architectural Boundary Invariant test.

Classification: CORE
Enforces that no file under the canonical `researchforge/` namespace ever imports
from the legacy top-level modules (agents, ecrm, evolution, rdg, policy, failure,
tools, config, benchmarks).
"""
from __future__ import annotations

import ast
from pathlib import Path
import pytest

LEGACY_NAMESPACES = frozenset([
    "agents",
    "ecrm",
    "evolution",
    "rdg",
    "policy",
    "failure",
    "tools",
    "config",
    "benchmarks",
])

# Find project root directory
PROJECT_ROOT = Path(__file__).parent.parent
CANONICAL_ROOT = PROJECT_ROOT / "researchforge"


def test_no_legacy_imports_in_canonical_namespace():
    """No file under researchforge/ may import from legacy top-level modules."""
    violations = []
    for py_file in CANONICAL_ROOT.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    continue  # relative import within researchforge is allowed
                module = node.module
            elif isinstance(node, ast.Import):
                module = ".".join(alias.name for alias in node.names)
            else:
                continue

            if module and module.split(".")[0] in LEGACY_NAMESPACES:
                rel_path = py_file.relative_to(PROJECT_ROOT)
                violations.append(f"{rel_path}:{getattr(node, 'lineno', '?')} imports {module!r}")

    assert not violations, (
        f"Cross-stack boundary violations found ({len(violations)}):\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


def test_no_database_imports_in_domain_contracts():
    """Domain contracts must never import SQLAlchemy, alembic, psycopg, or persistence modules."""
    forbidden_database_pkgs = frozenset(["sqlalchemy", "alembic", "psycopg", "psycopg2", "sqlite3"])
    domain_root = CANONICAL_ROOT / "domain"
    violations = []

    for py_file in domain_root.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules_to_check = []
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    modules_to_check.append(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    modules_to_check.append(alias.name)

            for mod in modules_to_check:
                pkg = mod.split(".")[0]
                if pkg in forbidden_database_pkgs or "repositories" in mod:
                    rel_path = py_file.relative_to(PROJECT_ROOT)
                    violations.append(f"{rel_path}:{getattr(node, 'lineno', '?')} imports {mod!r}")

    assert not violations, (
        f"Domain-layer database leakage found ({len(violations)}):\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


def test_no_database_imports_in_api_controllers():
    """API controllers and services must never import SQLAlchemy, alembic, psycopg, or internal SQL tables."""
    forbidden_database_pkgs = frozenset(["sqlalchemy", "alembic", "psycopg", "psycopg2", "sqlite3"])
    api_root = CANONICAL_ROOT / "api"
    violations = []

    for py_file in api_root.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules_to_check = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules_to_check.append(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    modules_to_check.append(alias.name)

            for mod in modules_to_check:
                pkg = mod.split(".")[0]
                if pkg in forbidden_database_pkgs:
                    rel_path = py_file.relative_to(PROJECT_ROOT)
                    violations.append(f"{rel_path}:{getattr(node, 'lineno', '?')} imports forbidden db package {mod!r}")
                if "sql.models" in mod or "sql.database" in mod:
                    rel_path = py_file.relative_to(PROJECT_ROOT)
                    violations.append(f"{rel_path}:{getattr(node, 'lineno', '?')} imports internal ORM model {mod!r}")

    assert not violations, (
        f"API-layer database leakage found ({len(violations)}):\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


def test_no_database_imports_in_statistical_benchmark():
    """Statistical evaluation package must never import SQLAlchemy, alembic, psycopg, or persistence models."""
    forbidden_database_pkgs = frozenset(["sqlalchemy", "alembic", "psycopg", "psycopg2", "sqlite3"])
    stat_root = CANONICAL_ROOT / "benchmarks" / "statistical"
    violations = []

    for py_file in stat_root.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules_to_check = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules_to_check.append(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    modules_to_check.append(alias.name)

            for mod in modules_to_check:
                pkg = mod.split(".")[0]
                if pkg in forbidden_database_pkgs or "repositories" in mod:
                    rel_path = py_file.relative_to(PROJECT_ROOT)
                    violations.append(f"{rel_path}:{getattr(node, 'lineno', '?')} imports forbidden db package {mod!r}")

    assert not violations, (
        f"Statistical package database leakage found ({len(violations)}):\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


def test_no_database_imports_in_cohort_benchmark():
    """Cohort specification package must never import SQLAlchemy, alembic, psycopg, or persistence models."""
    forbidden_database_pkgs = frozenset(["sqlalchemy", "alembic", "psycopg", "psycopg2", "sqlite3"])
    cohort_root = CANONICAL_ROOT / "benchmarks" / "cohort"
    violations = []

    for py_file in cohort_root.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules_to_check = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules_to_check.append(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    modules_to_check.append(alias.name)

            for mod in modules_to_check:
                pkg = mod.split(".")[0]
                if pkg in forbidden_database_pkgs or "repositories" in mod:
                    rel_path = py_file.relative_to(PROJECT_ROOT)
                    violations.append(f"{rel_path}:{getattr(node, 'lineno', '?')} imports forbidden db package {mod!r}")

    assert not violations, (
        f"Cohort package database leakage found ({len(violations)}):\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


def test_no_database_imports_in_execution_benchmark():
    """Execution benchmark package must never import SQLAlchemy, alembic, psycopg, or persistence models."""
    forbidden_database_pkgs = frozenset(["sqlalchemy", "alembic", "psycopg", "psycopg2", "sqlite3"])
    exec_root = CANONICAL_ROOT / "benchmarks" / "execution"
    violations = []

    for py_file in exec_root.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules_to_check = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules_to_check.append(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    modules_to_check.append(alias.name)

            for mod in modules_to_check:
                pkg = mod.split(".")[0]
                if pkg in forbidden_database_pkgs or "repositories" in mod:
                    rel_path = py_file.relative_to(PROJECT_ROOT)
                    violations.append(f"{rel_path}:{getattr(node, 'lineno', '?')} imports forbidden db package {mod!r}")

    assert not violations, (
        f"Execution package database leakage found ({len(violations)}):\n"
        + "\n".join(f"  - {v}" for v in violations)
    )




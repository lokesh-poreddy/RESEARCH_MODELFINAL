# ResearchForge-ECRM Integration Plan (v1.0.0)
**Date:** 2026-11-01 | **Status:** EXECUTED

## 1. Existing Architecture Summary
ResearchForge-ECRM is a two-axis research intelligence platform:
- **Axis A (Scientific Evidence):** Phase 1.x pipeline — domain extraction, VRDEG, SQLAlchemy repos, Alembic migrations 001–006
- **Axis B (Evolutionary ML):** RF-1 through RF-3 — ResearchController, contextual bandit, DSL-based algorithm discovery

Both axes share one canonical Python package (`researchforge/`) and are tested by one unified test suite.

## 2. Integration Map

| Component | Action | Outcome |
|---|---|---|
| RF-2 transfer modules | Bug-fix + upgrade | 3 bugs fixed, telemetry added |
| RF-3 discovery modules | Upgrade pilot evaluation | Real empirical scoring replaces mock |
| ContextualBandit | Extend with add_action() | Dynamic arm injection for RF-3 |
| operators.py | Add DYNAMIC_STRATEGIES registry | RF-3 operators become live choices |
| Benchmark scripts | Upgrade to production metrics | Running-best AUC, Cohen's d |
| researchforge/__init__.py | Version bump 0.1.0→1.0.0 | Reflects integrated state |
| CHANGELOG.md | Add RF-2 + RF-3 entries | Traceability restored |
| RELEASE_MANIFEST.json | Bump to 1.0.0 | Single source of truth |

## 3. Risk and Mitigation
- **Risk:** New DSL operators could produce degenerate genomes → **Mitigated** by safety bounds in DSLCompiler
- **Risk:** Version mismatch confusion → **Mitigated** by version bump + CHANGELOG update
- **Risk:** Legacy directories polluting imports → **Mitigated** by cross-stack import guard tests/test_no_cross_stack_imports.py

## 4. Rollback
Git ref `28fd0f7` is the pre-audit checkpoint on `rf-3-algorithm-discovery`. Revert with:
```
git revert HEAD..28fd0f7 --no-commit
```

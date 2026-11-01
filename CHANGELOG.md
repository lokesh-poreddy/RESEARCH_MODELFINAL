# ResearchForge-ECRM Changelog

All meaningful changes to the ResearchForge-ECRM system are documented here.
Failed experiments, negative results, and wrong decisions are **research evidence**
and must be preserved — they are not removed from this file.

Format follows [Keep a Changelog](https://keepachangelog.com).

---
## [1.0.0] — 2026-11-01 — RF-1 + RF-2 + RF-3 Integrated Release

**Character of release:** Production-hardened integration unifying three generations:
RF-1 (evolutionary controller), RF-2 (contextual transfer policy), and RF-3 (autonomous algorithm discovery).

### Added — RF-3: Algorithm Discovery Pipeline (Phase 13)
- **`researchforge/discovery/operator_genome.py`**: `OperatorGenome` evolutionary mutation; 3 seed operators.
- **`researchforge/discovery/dsl_compiler.py`**: 8 AST-to-genome primitive handlers with safety bounds.
- **`researchforge/discovery/meta_controller.py`**: Real two-arm pilot evaluation; auto-promotes to live bandit.
- **`researchforge/genome/operators.py`**: `DYNAMIC_STRATEGIES` registry + `register_dynamic_strategy()`.
- **`researchforge/policy/contextual_bandit.py`**: `add_action()` for dynamic arm injection.

### Added — RF-2: Contextual Transfer Policy (Phase 12)
- **`researchforge/transfer/utility_learner.py`**: Online Bayesian linear regression, 9 normalised features.
- **`researchforge/transfer/sampling.py`**: Counterfactual sampling, epsilon 20%→1% decay.
- **`researchforge/policy/utility_gate.py`**: Bayesian TRANSFER/SUPPRESS/ABSTAIN gating.

### Fixed — Bugs Resolved in Audit 2026-11-01
- `utility_learner.py`: Removed duplicate `return x`; log1p-normalised `observation_count`; beta 1→10.
- `sampling.py`: Division-by-zero guard; epsilon range fixed; trigger telemetry added.
- `operator_genome.py`: Primitive vocab synced with DSLCompiler.
- `meta_controller.py`: Real two-arm pilot replaces mock evaluator.
- `benchmark_b1_regression.py`: AUC now uses running-best curve. PASS/FAIL criterion added.
- `benchmark_b10_ablation.py`: Cohen's d effect sizes added.

### Validated Benchmark Results
- B1 Regression: RF-2 0.8779 ≥ 98% of RF-1 0.8825 → **PASS**
- B3 Transfer Matrix: T6 +0.352, T5 +0.042 (out-of-distribution benefit confirmed)
- B10 Ablation: Cohen's d vs Full RF-2 computed for all 4 conditions
- Import health: 11/11 canonical modules clean

---
## [RF-1.0.0-alpha.2.1] — 2026-09-03 — Architecture Convergence & Contract Hardening


**Character of release**: Architectural. Reconciles dual-stack architecture into
ONE canonical domain model (14 objects), ONE canonical execution path (TMG population),
ONE canonical provenance model (ExperimentSpec/Run/Outcome), ONE canonical configuration
hierarchy (4 scopes), and AST-enforced boundary isolation. 269 tests pass (171 CORE,
51 COMPATIBILITY, 47 LEGACY) with 0 failures. Trajectory bitwise identical to baseline.

### Added
- **14 Canonical Domain Objects**:
  - Class A immutable: `ResearchSystemGenome`, `TargetModelGenome`, `ExperimentSpec`,
    `ExperimentRun`, `ExperimentOutcome`, `Evidence`, `Artifact`, `Provenance`,
    `ResearchDecision`, `Failure`.
  - Class B versioned mutable: `MemoryRecord`.
  - Class C reserved positions: `ResearchProblem`, `Hypothesis`.
  - Class D binding: `ResearchState`.
- **Deterministic Trajectory Fingerprinting**:
  - `TrajectoryFingerprint` and `compute_trajectory_fingerprint` in `researchforge/experiment/trajectory.py`.
- **Cross-Stack Import Guard**:
  - `tests/test_no_cross_stack_imports.py` enforces zero legacy imports from canonical code.
- **Canonical Configuration Hierarchy**:
  - Scope 1 (Software): `researchforge/config/rf_config.py` (`RFConfig`).
  - Scope 2 (Environment): `researchforge/config/env_config.py` (`EnvConfig`).
  - Scope 3 (Research Policy): `RSG.execution_config` and sub-configs.
  - Scope 4 (Experiment): `ExperimentSpec`.
- **Automated Release Manifest**:
  - `scripts/capture_release_state.py` auto-counts tests by marker, extracts git status and package versions.
- **66 New CORE Tests**:
  - `test_genomes.py` (+5 tests for execution/termination configs)
  - `test_experiment.py` (20 tests for experiment domain objects)
  - `test_state.py` (8 tests for ResearchState and controller state integration)
  - `test_research.py` (6 tests for ResearchProblem and Hypothesis)
  - `test_canonical_invariants.py` (8 tests covering shared canonical invariants)
  - `test_no_cross_stack_imports.py` (1 test boundary verification)
  - `test_retrieval.py` (10 tests covering literature retrieval adapters)
  - `test_config.py` (8 tests covering configuration hierarchy)

### Changed
- `ResearchController`:
  - Population upgraded from `List[ModelGenome]` to `List[TargetModelGenome]`.
  - Integrates `ResearchState` generation into run loop (`result.states`).
  - Wires `RSG.memory_config` and `RSG.execution_config`.
  - Preserves bitwise-identical trajectories to baseline across seeds 0, 1, 2.
- `researchforge/genome/research_system_genome.py`:
  - Added `ExecutionConfig`, `OperatorConfig`, `TerminationConfig`.
- Legacy modules (`agents/`, `ecrm/`, `evolution/`, `rdg/`, `policy/`, `failure/`, `tools/`, `config/`):
  - Annotated with machine-readable `LEGACY_STATUS` dictionaries.

---

## [RF-1.0.0-alpha.2] — 2026-09-02 — Two-Level Genome Architecture

**Character of release**: Architectural. Introduces the RSG/TMG genome split —
the foundational research-artifact model for VRDEG (alpha.3+). No changes to
the research loop, algorithms, or existing tests. 105/105 tests pass.
Regression proof: 6 comparisons (3 seeds × 2 conditions), max delta = 0.0,
all trajectory hashes bitwise-identical.

### Added

**ResearchSystemGenome (RSG)** — `researchforge/genome/research_system_genome.py`
- Governs *how* ResearchForge conducts research: phase, budget, operator portfolio,
  memory policy, validity gate config, retrieval config, termination criteria
- Conceptual scope: *a versioned executable specification of the ResearchForge
  research policy*, not a configuration dump
- Strict JSON Schema (`additionalProperties: false`)
- `RSG.default(condition, seed)` — factory that codifies alpha.1 defaults as a
  versioned research artifact
- `RSG.evolve(operator, rng, child_index, mutation_params)` — immutability-contract
  pattern (returns new RSG, never mutates self)
- `RSG.fingerprint()` — sha256 of canonical JSON (excluding timestamp)
- `RSG.validate()`, `to_dict()`, `from_dict()`, `to_json()`
- `researchforge_version` field — RF release that created this RSG (AD-012)
- Separate `RSG_EVOLUTION_OPERATORS` namespace (distinct from `TMG_OPERATORS`)
- Full `evolve()` support for 11 operators: expand_budget, contract_budget,
  shift_phase, alter_exploration_constant, alter_memory_decay, reweight_operators,
  add_operator, remove_operator, crossover_policies, …

**TargetModelGenome (TMG)** — `researchforge/genome/target_model_genome.py`
- Governs *what* model is being optimized
- Backward-compatible superset of ModelGenome (all RF-0.x fields preserved)
- `from_model_genome(mg)` — lossless RF-0.x upgrade (preservation guarantee)
- `to_model_genome()` — downgrade for compatibility; roundtrip proven: `to_mg(from_mg(mg)) == mg`
- Deterministic, sibling-collision-safe `tmg_id` (AD-005): sha256 of
  `(parent_ids, operator, generation, seed, mutation_parameters, child_index)[:16]`
- Canonical serialization: `tmg_id` only (not `model_id`) — schema debt avoided (AD-007)
- `clone(operator, child_index)`, `clone_for_crossover()` — immutability contract
- `fingerprint()` — sha256 of canonical JSON
- `TMGCapabilities` dataclass: `supports_warm_start/partial_fit/predict_proba`,
  `expected_train_time_s`, `memory_estimate_mb`, `infer(model_type)` factory
- Extended lineage: `ancestor_ids`, `crossover_parents`, `rollback_from`
- `build_estimator()` delegates to ModelGenome for behavioral parity (AD-004 extension)
- `TMG_OPERATORS` namespace; `researchforge_version` field

**Shared schema utilities** — `researchforge/genome/schema.py`
- `deterministic_genome_id(prefix, parent_ids, operator, generation, seed,
  mutation_parameters, child_index)` — sibling-collision-safe, reproducible
- `genome_fingerprint(canonical_dict)` — sha256 hex digest of canonical JSON
- `validate_genome(d, schema)` — strict jsonschema wrapper with path-aware errors
- `GENOME_SCHEMA_VERSION_TMG = "1.0"`, `GENOME_SCHEMA_VERSION_RSG = "1.0"`

**Migration registry** — `researchforge/genome/migration.py`
- `migrate_tmg(d, target_version)` — accepts legacy `model_id` alias, produces
  canonical `tmg_id`; adds `schema_version`, `ancestor_ids`, `capabilities`
- `migrate_rsg(d, target_version)` — RSG migration
- `TMGMigrationRegistry` — (None, "1.0") migration registered
- Documented distinction: *preservation guarantee* vs *derived metadata*

**Regression benchmark** — `scripts/rf0_vs_rf1_regression.py`
- RDE-Bench over 3 seeds × 2 conditions for rsg=None vs rsg=RSG.default()
- Produces `RF0_vs_RF1_regression.json` with per-comparison metrics + trajectory hashes
- Fails with exit code 1 if `max_delta > 0.005`

**Test suite** — `tests/test_genomes.py` (28 tests)
- Tier 1 RSG (7): factory, validate, bad-phase rejection, determinism, JSON keys, evolve, roundtrip
- Tier 1 TMG (9): factory all families, validate, schema fields, pipeline equivalence,
  safety check equivalence, clone lineage, operator field, deterministic ID, sibling IDs
- Tier 2 migration (7): field preservation, model_genome roundtrip, legacy model_id,
  schema rejects leaked model_id, all 7 operators produce valid TMG, RSG roundtrip, v0→v1 migration
- Tier 3 integration (5): controller accepts RSG, rsg_id=None when not provided,
  **behavioral equivalence test** (trajectory hashes over 2 seeds), fingerprint stability × 2

**SVG-v1 documentation corrections** — `researchforge/scientific_validity/__init__.py`
- `SVG_V1_KNOWN_LIMITATIONS` machine-readable dict added to public API
- `BaselineFairnessValidator` conceptually re-labelled as TrivialBaselineCheck
- `LabelPermutationTest` re-labelled as LabelRandomisationSanityCheck
- `paired=True` default documented for RF-vs-RF comparisons
- `GATE_VERSION = "SVG-v1"` (renamed from "1.0.0-alpha.1")

### Modified

**`researchforge/pipeline/controller.py`**
- Added optional `rsg: Optional[Any] = None` parameter (AD-006, AD-013)
- `rsg=None` → exactly the same execution path as alpha.1 (proven, not assumed)
- `rsg=RSG.default(condition)` → stores RSG for provenance only; no behavior change
- `RunResult.rsg_id: Optional[str]` field added (None when rsg=None)

**`researchforge/genome/__init__.py`**
- Exports all alpha.2 types: TMG, RSG, schema utilities, migration functions
- All RF-0.x exports unchanged

### Design Decisions

| Decision | Record |
|---|---|
| AD-005 | Deterministic sibling-safe TMG IDs |
| AD-006 | RSG optional, rsg=None = exact alpha.1 |
| AD-007 | tmg_id canonical; model_id compat-loader only |
| AD-008 | Preservation guarantee + derived metadata |
| AD-009 | Immutability via evolve/clone pattern |
| AD-010 | Separate operator namespaces (RSG vs TMG) |
| AD-011 | RSG = versioned research policy spec |
| AD-012 | schema_version ≠ researchforge_version |
| AD-013 | RSG behavioral invariant proven by benchmark |

### Explicitly NOT in this release (by design)

VRDEG, Adaptive-ECRM, Mode Router, Evidence Adjudication, SVG-v2
leakage checks, `PROVISIONAL` verdict status, pgvector/Neo4j backends,
real LLM research loop. Each is scheduled for its designated future release.

---

## [RF-1.0.0-alpha.1] — 2026-09 — Infrastructure Release

**Character of release**: Infrastructure only. Zero changes to research algorithms.
Behavioral equivalence with RF-0.x is verified by the regression harness (8 tests
including re-execution of all 17 original tests, and exact numeric parity checks
for ECRM scoring algorithms).

### Added

**Adapter Layer** (`researchforge/adapters/`)
- `errors.py` — `BackendError` hierarchy: `BackendConnectionError`,
  `BackendNotFoundError`, `BackendSerializationError`, `BackendTransactionError`,
  `BackendCapabilityError`, `DimensionMismatchError`.
- `capabilities.py` — `BackendCapabilities`, `BackendInfo`, `HealthStatus`.
- `protocols.py` — `GraphBackend` and `VectorIndexBackend` Protocol base classes
  with complete docstring-level contracts (transaction protocol, dimension contract,
  metadata support). `CRITICAL RULE (AD-003)`: GraphBackend has NO knowledge of
  RDG semantic constraints. Domain validation stays in `rdg/graph.py`.
- `backends/memory.py` — `InMemoryGraphBackend`: zero-dep, transactional
  (copy-on-write rollback), health-check.
- `backends/sqlite.py` — `SQLiteGraphBackend`: WAL mode, persistent, full
  BEGIN/COMMIT/ROLLBACK transaction support. Current substitute for Neo4j.
- `backends/neo4j.py` — `Neo4jGraphBackend`: **documented production stub**.
  Raises `NotImplementedError` on instantiation. Includes the complete 9-step
  migration path to a live Neo4j instance.
- `backends/inprocess_vector.py` — `InProcessVectorIndex`: brute-force exact
  cosine similarity, dimension-contracted, metadata-aware. Replaces the ad-hoc
  inline cosine loop previously in `ecrm.py`.
- `backends/pgvector.py` — `PgvectorBackend`: **documented production stub**.
  Raises `NotImplementedError`. Includes complete PostgreSQL+pgvector migration path.
- `registry.py` — `AdapterRegistry` with `register_graph_backend()`,
  `create_graph_backend()`, `register_vector_backend()`, `create_vector_backend()`,
  `*_info()`, `list_*()`. Pre-populated default registry with `in_memory`, `sqlite`
  (graph), `inprocess` (vector). Stubs intentionally not registered.
- `validation.py` — Three-level validation harness: Level 1 Contract
  (CRUD + transaction lifecycle + error types), Level 2 Persistence
  (data survives close/reopen), Level 3 Integration (full RDG chain on
  top of backend + typed-edge constraint enforcement). `validate_graph_backend()`
  and `validate_vector_backend()` functions.

**Scientific Validity Gate** (`researchforge/scientific_validity/`)
- `verdicts.py` — `ValidityVerdict` (PASS/FAIL/WARNING/INCONCLUSIVE/
  REQUIRES_HUMAN_REVIEW), `ClaimEligibility`, `CheckSeverity`, `CheckResult`,
  `ValidityReport.finalize()` with correct blocker/warning aggregation.
- `leakage.py` — `DataLeakageDetector`: 4 checks: (A) exact duplicate samples,
  (B) near-duplicate (O(n²), optional), (C) target leakage (feature-label
  correlation), (D) index set overlap.
- `permutation.py` — `LabelPermutationTest`: permutes only training labels;
  tests against real test labels; reports `gap = real_metric - mean(perm_metrics)`.
- `baseline.py` — `BaselineFairnessValidator`: 6 checks: split size match,
  feature shape match, preprocessing match, training budget ratio, seed match,
  baseline above chance.
- `significance.py` — `StatisticalSignificanceTester`: paired t-test or
  Welch's unpaired; p-value approximation; Cohen's d effect size; 95%
  confidence intervals; correct PASS/FAIL/WARNING/INCONCLUSIVE verdict mapping.
- `gate.py` — `ScientificValidityGate`: orchestrates all checks, individual
  entry points (`run_leakage_check`, `run_permutation_check`,
  `run_significance_check`) and `run_standard_suite()`.
- `report.py` — `ValidityReportRenderer`: text (with icons, severity, blockers)
  and JSON renderers.

**Test suites**
- `tests/test_adapters.py` — 30 tests: Tier 1 Unit, Tier 2 Contract
  (validation harness), Tier 3 Integration (RDG + ECRM behavioral equivalence).
- `tests/test_scientific_validity.py` — 22 tests: all 4 check types, full gate
  integration, verdict aggregation, report rendering.
- `tests/test_regression.py` — 8 tests: backward API parity, typed-edge
  enforcement, evidence chain correctness, ECRM algorithm parity (exact numeric),
  ModelGenome pipeline parity, controller smoke (all 4 conditions), 17 original
  tests re-run via subprocess.

**Project state**
- `RESEARCHFORGE_STATE.yaml` — Complete project memory checkpoint: release
  manifest, implementation inventory (implemented/planned), benchmark history
  (RF-0.x digits and ECG results), research record (questions, hypotheses,
  findings, negative results), architectural decisions (AD-001 to AD-004),
  verification matrix, reproducibility, release history.

### Changed

**RDG (`researchforge/rdg/graph.py`)** — backward-compatible refactoring
- Added `backend=` constructor parameter (opt-in, RF-1.0+).
- Original `db_path=` API fully preserved and unchanged.
- `add_node()` and `add_edge()` now route persistence through
  `self._backend` instead of calling `_con.execute()` directly.
- `EdgeConstraintError` and all typed-relation rules remain in this class —
  not in the backend (AD-003).
- Added `_reload_from_backend()` for reconstructing the in-process traversal
  index from an existing persistent backend after process restart.
- Added `close()` that closes the backend.

**ECRM (`researchforge/memory/ecrm.py`)** — behavioral-equivalence refactoring
- `vector_backend=` constructor parameter added (opt-in, RF-1.0+).
- Original `db_path=` API fully preserved.
- Vector similarity search delegated to `VectorIndexBackend.search()` and
  `VectorIndexBackend.add()` instead of the inline cosine loop.
- `store()` now calls `self._vector_backend.add(rec.id, embedding, metadata=...)`.
- `query()` calls `self._vector_backend.search(qvec, k=None)` then applies
  archived-record filtering in Python.
- **ALL SCORING ALGORITHMS UNCHANGED** (AD-004): `strategy_stats()`,
  `negative_transfer_rate()`, `research_experience_score()`, `consolidate()`,
  `working_memory()`, `long_term_memory()`, `reallocate()`,
  `memory_half_life_days()`, `has_similar_failure()`.

### Preserved (from RF-0.x)

These are permanent research artifacts. They must not be altered:

- The trajectory-memory negative result (NR-001): trajectory-conditioned
  memory did NOT improve over flat ECRM at 25-generation budget on either
  task. FRR 0.63 (digits), 0.41 (ECG), same for both conditions.
- The RF-0.x benchmark results in `RESEARCHFORGE_STATE.yaml`.
- All 17 original tests (`tests/test_basic.py`).

### Known Limitations (unchanged from RF-0.x)

- SafeRunner does NOT provide filesystem/network isolation or cgroup limits.
- ECRM uses hashed bag-of-words embedding, not a semantic encoder (AD-002).
- Neo4j and pgvector backends are documented stubs (`NotImplementedError`).

### Acceptance Criteria — VERIFIED

| Criterion | Status |
|---|---|
| 17 RF-0.x tests pass | ✅ 17/17 |
| 30 adapter tests pass (Tier 1+2+3) | ✅ 30/30 |
| 22 validity gate tests pass (Tier 4) | ✅ 22/22 |
| 8 regression tests pass (Tier 5) | ✅ 8/8 |
| ECRM behavioral equivalence (exact numeric) | ✅ verified |
| RDG backward compat (db_path=) | ✅ verified |
| Neo4j stub: NotImplementedError | ✅ verified |
| pgvector stub: NotImplementedError | ✅ verified |
| Stubs not in default registry | ✅ verified |
| RESEARCHFORGE_STATE.yaml created | ✅ |
| NR-001 negative result preserved | ✅ in STATE.yaml |
| CHANGELOG.md | ✅ this file |

---

## [RF-0.x] — 2026-09 — Verified Baseline

**Character of release**: Reference implementation. The permanent experimental
control against which all future RF versions are compared.

### Implemented

- `ResearchDevelopmentGraph` (RDG): typed in-memory graph with SQLite mirror.
  10 node types, 10 edge relations, `EDGE_TYPE_CONSTRAINTS`, `evidence_chain()`.
- `ModelGenome` + 7 evolution operators (increase_capacity, decrease_capacity,
  change_family, tune_hyperparameters, change_pipeline, crossover_top2,
  mutate_random). Real sklearn Pipeline via `build_estimator()`.
- `ECRM`: `store()`, `query()`, `has_similar_failure()`, `strategy_stats()`,
  `negative_transfer_rate()`, `research_experience_score()`, `consolidate()`,
  `working_memory()`, `long_term_memory()`, `reallocate()`,
  `memory_half_life_days()`.
- `TrajectoryMemory`: context-conditioned on (strategy, model_type,
  capacity_bucket). RDG provenance links.
- `UCB PolicyLearner`.
- Rule-based failure diagnosis (5 categories).
- `SafeRunner` with process-level kill.
- `ResearchController` + pipeline (4 conditions).
- RDE-Bench, digits_task, synthetic_ecg_task.
- FastAPI REST layer (`/run`, `/benchmark`, `/memory`, `/rdg`, `/health`).
- GitHub retrieval (real GitHub API).
- MATLAB export (one-way, visualization only).
- SQLite persistence.

### Key Finding (NR-001)

Trajectory-memory (conditioned on strategy + model_type + capacity_bucket)
did NOT improve over flat ECRM at 25-generation budget:
- digits: full=0.870, trajectory=0.8707, no_memory=0.8705. FRR 0.63 for both.
- ECG: full=0.722, trajectory=0.708, no_memory=0.707. FRR 0.41 for both.

**Root cause hypothesis**: context fragmentation. Too many sparse evidence
bins at 25-generation budget. Motivates adaptive backoff (RF-1.0.0-alpha.4).

### Tests

17 tests. All pass.

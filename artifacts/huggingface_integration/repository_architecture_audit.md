# ResearchForge-ECRM — Repository Architecture Audit

**Audit date:** 2026-09-06  
**Method:** Read-only inspection of executable source, tests, schemas, and tracked state files. Comments were treated as claims and verified against code.  
**Git worktree at audit start:** modified `researchforge/benchmarks/execution/engine.py`, `researchforge/benchmarks/execution/protocol.py`; untracked `test_debug.py`, `tests/test_phase12b_6_freeze.py`.

This document is an inventory, not an implementation claim. Hugging Face was **not** present in the canonical package at audit time.

---

## 1. Root project structure

| Path | Role |
|---|---|
| `researchforge/` | Canonical package. All new research-system work belongs here. |
| `agents/`, `ecrm/`, `evolution/`, `rdg/`, `policy/`, `failure/`, `tools/`, `config/`, top-level `benchmarks/` | Legacy/compatibility stack. `tests/test_no_cross_stack_imports.py` forbids canonical code from importing these namespaces. |
| `tests/` | CORE / COMPATIBILITY / LEGACY pytest suite (`pytest.ini` markers). |
| `scripts/` | Release, regression, Phase 12B preflight, RF-FLEX, MATLAB evolution reports. |
| `docs/` | Architecture audit, literature positioning, v2 protocol. |
| `artifacts/` | Generated evidence (including this directory). |
| `schemas/`, `researchforge/schemas/` | JSON schemas (RDG/VRDEG/domain). |
| Root JSON (`phase6_*.json`, `phase12b_*.json`, `RF0_vs_RF1_regression.json`, …) | Frozen or generated protocol/benchmark artifacts. Do not rewrite as part of provider integration. |
| `RESEARCHFORGE_STATE.yaml` | Machine-readable project memory. **Lagging** relative to source: still lists VRDEG, RSG wiring, Adaptive-ECRM, router, etc. as planned while those modules exist and have tests. |
| `README.md` | Accurate for RF-0.x / early RF-1 narrative; incomplete for Phases 8–13. |
| `CHANGELOG.md` | Stops at RF-1.0.0-alpha.2.1; later phases exist in code/tests but are not fully changelogged here. |
| `.env.example` | Legacy LLM keys (OpenAI/Anthropic), DB, OpenAlex, GitHub. No `HF_TOKEN`. |
| `requirements.txt` | numpy, sklearn, jsonschema, fastapi, uvicorn, pydantic, httpx, matplotlib, sqlalchemy, alembic, psycopg2-binary. **No** `huggingface_hub`, openai, anthropic, sentence-transformers. `requests` is imported by OpenAlex retriever but not pinned. |

---

## 2. Subsystem inventory

Status vocabulary matches later capability matrix: IMPLEMENTED / PARTIAL / STUB / UNAVAILABLE / LEGACY / PLANNED.

### Canonical researchforge/ package

**Purpose:** Single-machine, reproducible research-development framework: evolve sklearn model genomes, remember outcomes, gate scientific claims, persist provenance.  
**Primary source:** `researchforge/__init__.py` (`__version__ = "0.1.0"` — version string is stale vs RF-1.0.0-alpha.2.1 in `RFConfig`).  
**Public interfaces:** Subpackages (`genome`, `memory`, `pipeline`, `rdg`, `vrdeg`, `policy`, `adapters`, `api`, …).  
**Status:** IMPLEMENTED (core loop) + many later-phase modules.  
**Tests:** `tests/test_*.py` (60+ modules).  
**Dependencies:** Local scientific stack; optional network for retrieval.  
**External services:** None required for the controller loop.  
**Limitations:** Not a globally connected autonomous lab; sklearn TMG families only; hashed embeddings; no canonical LLM.  
**Research significance:** The executable substrate against which all provider integrations must remain opt-in.

---

### Legacy/compatibility stack

**Purpose:** Earlier dual-stack (agents + OpenAI/Anthropic/Gemini, top-level ECRM, evolution). Preserved as research evidence.  
**Primary source:** `agents/base_agent.py`, `config/settings.py`, `ecrm/`, `evolution/`, `failure/`, `tools/`.  
**Public interfaces:** `BaseAgent.llm_call`; `Settings` via pydantic-settings.  
**Status:** LEGACY. LLM providers are real code paths but **outside** the canonical import boundary.  
**Tests:** LEGACY/COMPATIBILITY marked tests (`test_ecrm.py`, `test_evolution.py`, `test_failure.py`, `test_policy.py`, `test_integration.py`).  
**Dependencies:** Optional `openai`, `anthropic`, `google.generativeai`, `sentence_transformers` (none in `requirements.txt`).  
**External services:** OpenAI / Anthropic / Gemini if keys present; otherwise deterministic mock.  
**Limitations:** Cannot be used from `researchforge/` without violating the AST boundary test.  
**Research significance:** Shows a previous LLM design; **not** the integration point for Hugging Face.

---

### Domain contracts

**Purpose:** Class A–D canonical objects (problems, hypotheses, decisions, evidence, genomes, experiments, state, provenance, artifacts).  
**Primary source:** `researchforge/domain/`.  
**Public interfaces:** Frozen/typed dataclasses + `to_dict`/`from_dict`. Domain must not import SQLAlchemy (`test_no_cross_stack_imports.py`).  
**Status:** IMPLEMENTED.  
**Tests:** `tests/test_domain_contracts.py`, `tests/test_domain_schemas.py`, `tests/test_canonical_invariants.py`.  
**Dependencies:** None external.  
**External services:** None.  
**Limitations:** Dual TMG types exist (`researchforge.genome.target_model_genome.TargetModelGenome` vs `researchforge.domain.genome`); repositories use domain genomes.  
**Research significance:** Provider results must become domain evidence/artifacts, not raw vendor JSON.

---

### Schemas

**Purpose:** Strict JSON Schema (`additionalProperties: false`) for genomes, RDG, VRDEG, diagnosis, decisions.  
**Primary source:** `researchforge/genome/schema.py`, `researchforge/rdg/schema.py`, `researchforge/schemas/`, `schemas/rdg_schema.json`.  
**Status:** IMPLEMENTED.  
**Tests:** Genome/domain schema tests.  
**Limitations:** TMG `model_type` enum is sklearn-only (`MLPClassifier`, `RandomForestClassifier`, `SVC`, `LogisticRegression`).  
**Research significance:** HF Hub IDs cannot replace TMG identity; they must attach as optional origin metadata or fail schema validation.

---

### ResearchState

**Purpose:** Generation-t snapshot binding RSG, TMG ids, metrics, evidence, budget.  
**Primary source:** `researchforge/state/research_state.py` (controller-facing) and `researchforge/domain/state.py` (transition engine).  
**Public interfaces:** `ResearchState`, `to_dict`/`from_dict`/`fingerprint`.  
**Status:** IMPLEMENTED (two related types — controller uses package `state.research_state`).  
**Tests:** `tests/test_state.py`, `tests/test_state_transitions.py`.  
**Limitations:** Dual representations; transition engine does not drive `ResearchController.run()`.  
**Research significance:** Provider calls must not mutate historical states in place.

---

### ResearchStateTransitionEngine

**Purpose:** Event-sourced, non-mutating state transitions with provenance.  
**Primary source:** `researchforge/state/transition_engine.py`, `researchforge/state/events.py`, `researchforge/state/validators.py`.  
**Public interfaces:** `transition(state, event, provenance)`, `apply_events`.  
**Status:** IMPLEMENTED, not wired as the main controller loop.  
**Tests:** `tests/test_state_transitions.py`.  
**External services:** None.  
**Limitations:** Controller still appends `ResearchState` snapshots independently.  
**Research significance:** Correct place to record “HF discovery occurred” as events later; not required for first adapter.

---

### Target Model Genome (TMG)

**Purpose:** Versioned executable specification of the *model being optimized*.  
**Primary source:** `researchforge/genome/target_model_genome.py`.  
**Public interfaces:** `TargetModelGenome.default/from_model_genome/to_model_genome/clone/validate/build_estimator/safety_check/fingerprint`.  
**Status:** IMPLEMENTED for sklearn families.  
**Tests:** `tests/test_genomes.py`.  
**Limitations:** No Hub model representation; `build_estimator()` always sklearn.  
**Research significance:** Open-world HF candidates must be TMG-typed with explicit non-executability until safety/materialization, without changing sklearn semantics.

---

### Research System Genome (RSG)

**Purpose:** Versioned specification of *how* ResearchForge researches (budget, operators, memory, validity, execution).  
**Primary source:** `researchforge/genome/research_system_genome.py`.  
**Public interfaces:** `RSG.default/evolve/fingerprint`, `ExecutionConfig`, memory/validity configs.  
**Status:** IMPLEMENTED. Controller wires memory decay and sandbox timeout; validity gate params historically deferred.  
**Tests:** `tests/test_genomes.py`.  
**Limitations:** Not a full LLM/provider policy object.  
**Research significance:** Future `llm.provider=huggingface` belongs in env/software config, not silently inside RSG defaults (would change fingerprints).

---

### ECRM

**Purpose:** Evidence- and outcome-conditioned memory: store/query, RES, NTR, consolidate, forgetting.  
**Primary source:** `researchforge/memory/ecrm.py`, `researchforge/memory/record.py`, `researchforge/memory/embeddings.py`.  
**Public interfaces:** `ECRM.store/query/has_similar_failure/strategy_stats/...`.  
**Status:** IMPLEMENTED. Vector search via `VectorIndexBackend` (in-process cosine). Embeddings: `HashingVectorizer` 256-d bag-of-words (`AD-002`).  
**Tests:** `tests/test_adapters.py` (behavioral equivalence), CORE memory tests.  
**Dependencies:** sklearn HashingVectorizer; optional SQLite.  
**External services:** None.  
**Limitations:** Lexical not semantic similarity. Changing the embedder **would invalidate RF-0.x memory experiments**.  
**Research significance:** HF embeddings must be opt-in; default ECRM scoring must not change.

---

### Trajectory-ECRM / Adaptive Trajectory-ECRM

**Purpose:** Context-conditioned memory; adaptive backoff when bins are sparse (NR-001).  
**Primary source:** `researchforge/memory/trajectory.py`, `researchforge/memory/adaptive_trajectory.py`.  
**Status:** IMPLEMENTED; controller conditions `trajectory_memory`, `adaptive_trajectory`, `continuous_experience`, `cold_start`.  
**Tests:** `tests/test_adaptive_memory.py`, controller tests.  
**Research significance:** Negative result NR-001 is preserved evidence; do not “fix” it with an LLM.

---

### VRDEG

**Purpose:** Versioned research development and evidence graph (nodes, edges, lineage, canonical export).  
**Primary source:** `researchforge/vrdeg/graph.py`, `node.py`, `edge.py`, `queries.py`, `projector.py`.  
**Status:** IMPLEMENTED in-process.  
**Tests:** `tests/test_phase5_vrdeg.py`, `tests/test_vrdeg_queries.py`, `tests/test_phase5a_projector.py`.  
**Limitations:** Not Neo4j; not the storage backend for the sklearn controller RDG.  
**Research significance:** Provider provenance can be attached as nodes later; first HF slice should not rewrite VRDEG semantics.

---

### Evidence pipeline / adjudication

**Purpose:** Normalize experiments, negatives, literature into typed Evidence; store with contradiction preservation; snapshots for temporal isolation.  
**Primary source:** `researchforge/evidence/normalizer.py`, `store.py`, `snapshot.py`, `evidence.py`.  
**Public interfaces:** `EvidenceNormalizer`, `EvidenceStore`, `EvidenceSnapshot`, `EvidenceCandidate`.  
**Status:** IMPLEMENTED.  
**Tests:** `tests/test_evidence_pipeline.py`.  
**Limitations:** Literature hits are candidates, not adjudicated truth. Hub model cards must be treated the same way.  
**Research significance:** HF metadata = evidence with provenance, not scientific fact.

---

### Research Policy

**Purpose:** Transparent scored action selection with ablations, transfer classification, deterministic fingerprints.  
**Primary source:** `researchforge/policy/research_policy.py`, `score_decomposition.py`, `config.py`, `gating.py`.  
**Also:** UCB `policy_learner.py` used by `ResearchController` (the benchmarked loop).  
**Status:** IMPLEMENTED (two policies: UCB in the loop; ResearchPolicy for Phase 9).  
**Tests:** `tests/test_phase9_policy.py`.  
**Research significance:** Do not inject HF logits into UCB Q-values.

---

### Research Portfolio / Saturation

**Purpose:** Multi-branch search; saturation detection without auto-termination.  
**Primary source:** `researchforge/policy/portfolio.py`, `policy/saturation.py`.  
**Status:** IMPLEMENTED.  
**Tests:** `tests/test_phase9_portfolio.py`, `tests/test_phase9_saturation.py`.  
**External services:** None.

---

### TransferGuard

**Purpose:** Gate memory transfer across tasks (STRICT/SELECTIVE/DISABLED).  
**Primary source:** `researchforge/transfer/guard.py`.  
**Status:** IMPLEMENTED; opt-in on controller via `transfer_guard_mode`.  
**Tests:** `tests/test_phase13_transfer_guard.py`.

---

### DynamicModeRouter

**Purpose:** Deterministic mode machine (DEEP_RESEARCH, EXPERIMENTAL, REPLICATION, …).  
**Primary source:** `researchforge/router/router.py`, `modes.py`, `transitions.py`.  
**Status:** IMPLEMENTED; opt-in `enable_dynamic_router`.  
**Tests:** `tests/test_phase13_dynamic_router.py`.

---

### Scientific Validity Gate

**Purpose:** Leakage, permutation sanity, trivial baseline, paired significance.  
**Primary source:** `researchforge/scientific_validity/`.  
**Public interfaces:** `ScientificValidityGate.run_standard_suite`.  
**Status:** IMPLEMENTED (SVG-v1; known limitations documented in STATE.yaml).  
**Tests:** `tests/test_scientific_validity.py`.  
**Limitations:** Not publication-grade permutation/fairness; listed missing checks in STATE.yaml.  
**Research significance:** Must not be modified for HF. Hub claims do not pass the gate.

---

### Experiment lifecycle

**Purpose:** Spec → run → outcome with fingerprints; canonical runner + sklearn evaluator.  
**Primary source:** `researchforge/experiment/spec.py`, `run.py`, `outcome.py`, `runner.py`, `evaluators/sklearn_evaluator.py`.  
**Status:** IMPLEMENTED.  
**Tests:** `tests/test_experiment.py`, `tests/test_phase3_experiments.py`, `tests/test_phase4_execution_safety.py`.  
**Limitations:** Default executor is sklearn `fit/predict`. Hosted HF inference is a different experiment class.

---

### SafeRunner

**Purpose:** Subprocess timeout/kill and resource budget.  
**Primary source:** `researchforge/safety/sandbox.py`.  
**Status:** IMPLEMENTED.  
**Tests:** `tests/test_basic.py`, phase 4 safety tests.  
**Limitations:** No filesystem/network isolation, no cgroups (documented).  
**Research significance:** HF HTTP calls must not run inside the experiment subprocess as a hidden network channel for sklearn trials.

---

### Persistence (PostgreSQL / SQLite / in-memory)

**Purpose:** Unit of Work + repositories; SQLite/in-memory implementations; SQLAlchemy models; Alembic.  
**Primary source:** `researchforge/repositories/`, `researchforge/db/`, `alembic.ini`.  
**Status:** IMPLEMENTED (in-memory + SQL); Postgres schema exists (`db/schema.sql`).  
**Tests:** `tests/test_phase9_persistence.py`, `tests/test_persistence_backend.py`.  
**Stubs:** Neo4j graph backend, pgvector vector backend (`NotImplementedError` on construct).  
**External services:** Optional Postgres. Core tests use in-memory/SQLite.

---

### Execution manifest / engine / checkpoint / attestation / freeze / monitoring

**Purpose:** Phase 12B authorized 32,400-trial protocol: cryptographic binding, DAG, ledger, freeze.  
**Primary source:** `researchforge/benchmarks/execution/{manifest,engine,protocol,freeze,preflight,materialization}.py`, `monitoring/`.  
**Status:** IMPLEMENTED. Engine refuses execution unless attestation `execution_authorized`.  
**Tests:** `tests/test_phase12b_*.py`.  
**Limitation / invariant:** Integration must **not** authorize execution or rewrite frozen artifacts.  
**Research significance:** HF is orthogonal to confirmatory statistics.

---

### RF-FLEX

**Purpose:** Confirmatory / execution protocol for RF-FLEX releases.  
**Primary source:** `researchforge/benchmarks/rf_flex.py`, `rf_flex_execution.py`, `rf_flex_confirmatory.py`.  
**Tests:** `tests/test_phase13_rf_flex.py`, `tests/test_phase13_9_protocol.py`, `tests/test_phase13_10_execution.py`.  
**Status:** IMPLEMENTED; authorization state must remain unchanged.

---

### Statistical analysis plan / engine

**Purpose:** Sealed SAP; statistical engine for continuity hypotheses.  
**Primary source:** `researchforge/benchmarks/statistical/plan.py`, engine modules, `continuity/evaluator.py`.  
**Tests:** `tests/test_phase12_statistical_plan.py`, `tests/test_phase12_statistical_engine.py`, `tests/test_phase11_continuity.py`.  
**Status:** IMPLEMENTED and frozen conceptually. Do not alter.

---

### Existing APIs

**Purpose:** FastAPI: canonical research endpoints + operational `/run`-style leftovers.  
**Primary source:** `researchforge/api/server.py`, `dtos.py`, `services.py`. Dual `api/main.py` at repo root (legacy).  
**Status:** IMPLEMENTED.  
**Tests:** `tests/test_phase10_api.py`.  
**Limitations:** No HF routes.

---

### Existing adapter interfaces

**Purpose:** Storage adapters only: `GraphBackend`, `VectorIndexBackend`, registry, validation harness, Phase 10 `CanonicalAdapterManifest`.  
**Primary source:** `researchforge/adapters/protocols.py`, `registry.py`, `manifest.py`, `backends/`.  
**Status:** IMPLEMENTED for storage; **no** Model/Dataset/Inference/Embedding *provider* protocols at audit time.  
**Tests:** `tests/test_adapters.py`.  
**Stubs:** Neo4j, pgvector.  
**Research significance:** HF must **not** be jammed into GraphBackend. New provider protocols are required.

---

### Current LLM / provider abstraction

**Canonical:** `Synthesizer` protocol + `HeuristicSynthesizer` (deterministic operators). `LLMSynthesizer` raises `NotImplementedError` in `__init__` (`researchforge/pipeline/discovery.py`). Controller hard-codes `HeuristicSynthesizer()`.  
**Legacy:** `agents/base_agent.py` OpenAI/Anthropic/Gemini/mock.  
**Status:** STUB (canonical LLM); LEGACY (agent LLM).  
**How the controller obtains reasoning/planning/hypothesis/code generation today:** it does **not**. Strategy selection is UCB/random; genome edits are operator maps; diagnosis is rule-based; hypotheses in `researchforge/research/` are domain objects, not LLM-generated in the loop.

---

### Current embedding / vector abstraction

**Embedding function:** `researchforge/memory/embeddings.py::embed` (local hashing).  
**Vector index:** `VectorIndexBackend` (similarity search of *already computed* vectors).  
**Legacy:** `ecrm/embedder.py` optionally sentence-transformers.  
**Status:** IMPLEMENTED (local hashing) + storage protocol. No EmbeddingProvider.

---

### Model / dataset discovery

**Status:** UNAVAILABLE in canonical code. TMG population is `ModelGenome.default` families. Tasks are `digits_task` / `synthetic_ecg_task` (and related benchmark tasks). No Hub search.

---

### GitHub integration

**Purpose:** Repository search as optional retrieval enrichment.  
**Primary source:** `researchforge/retrieval/literature.py::GitHubRepositoryRetriever`.  
**Status:** IMPLEMENTED; unauthenticated; empty list on failure.  
**Also:** OpenAlex, Semantic Scholar, arXiv retrievers (`retrieve_all` in `retrieval/__init__.py`).  
**Tests:** `tests/test_retrieval.py`.  
**External services:** Optional HTTP. Core loop does not call retrieval every generation.

---

### Documentation and state files

| File | Audit note |
|---|---|
| `README.md` | RF-0.x honest simplifications; LLMSynthesizer unwired. |
| `CHANGELOG.md` | Through alpha.2.1. |
| `RESEARCHFORGE_STATE.yaml` | Baseline + alpha.1/2/2.1; next_upgrade still alpha.3 though VRDEG exists. |
| `docs/ALPHA3_ARCHITECTURE_AUDIT.md` | Architecture review. |
| `docs/LITERATURE_POSITIONING.md`, `docs/V2_RESEARCH_PROTOCOL.md` | Research framing. |
| Frozen JSON at repo root | Protocol evidence; do not mutate. |

---

### scripts/

Release seals, RF0 vs RF1 regression, Phase 12B preflight, RF-FLEX, overall evolution markdown generation. None talk to Hugging Face.

---

### tests/

CORE tests auto-marked in `conftest.py`. Markers CORE/COMPATIBILITY/LEGACY. No live-network requirement for the default suite.

---

## 3. How the ResearchController currently obtains capabilities

Verified in `researchforge/pipeline/controller.py`:

| Capability | Mechanism | External API |
|---|---|---|
| Reasoning | None (no LLM) | No |
| Hypothesis generation | RDG nodes + domain objects; not LLM | No |
| Planning | UCB `PolicyLearner` over `STRATEGIES` or random | No |
| Diagnosis | Rule-based `diagnose()` | No |
| Evidence processing | ECRM/trajectory store of trial outcomes | No |
| Embeddings | HashingVectorizer via `embed()` | No |
| Model candidates | Initial sklearn TMG + evolution operators | No |
| Datasets | In-process `Task` (sklearn digits / synthetic ECG) | No |
| Experiment execution | `evaluate_genome` ± SafeRunner | No |
| Code generation | None | No |
| Model evaluation | sklearn metric on held-out split | No |

---

## 4. Architectural constraints for Hugging Face (from this audit)

1. Implement inside `researchforge/adapters/` as **provider** protocols, not GraphBackend, not `agents/`.  
2. Keep `HeuristicSynthesizer` default; LLM is opt-in.  
3. Do not change ECRM default embedder.  
4. Do not authorize Phase 12B / RF-FLEX.  
5. Do not put secrets in artifacts, tests, or STATE.yaml.  
6. Offline pytest must pass without `HF_TOKEN`.  
7. TMG sklearn enum and fingerprints of default genomes must remain stable when origin is absent.

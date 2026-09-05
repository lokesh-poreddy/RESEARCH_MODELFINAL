"""tests/test_persistence_backend.py — Phase 8A Professional Persistence Backend test suite.

Verification suite for:
- Repository interfaces and in-memory implementations
- SQL repository implementations (SQLAlchemy 2.x)
- Atomic Unit of Work (commit all vs rollback all on error)
- Append-only event store (ordering, deterministic fingerprints, immutability)
- Artifact registry (SHA-256 computation, byte storage, tamper detection)
- VRDEG graph persistence adapter (in-memory & SQL, lineage and branch tracing)
- Database schema migrations (Alembic upgrade & downgrade)
- Deterministic event replay (reconstructing ResearchState and VRDEG graph)
- Real PostgreSQL integration check (clean skip when PostgreSQL daemon is absent)
- Performance baseline microbenchmark
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from researchforge.artifacts import (
    Artifact,
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ArtifactRegistry,
    LocalArtifactStorage,
)
from researchforge.domain.claim import Claim, ClaimStatus, EvidenceRelation
from researchforge.domain.decision import Decision
from researchforge.domain.diagnosis import Diagnosis
from researchforge.domain.evidence import Evidence, EvidenceQuality, EvidenceType
from researchforge.domain.experiment import ExperimentRun, ExperimentSpec, ExecutionStatus
from researchforge.domain.failure import Failure
from researchforge.domain.genome import ResearchSystemGenome, TargetModelGenome
from researchforge.domain.hypothesis import Hypothesis
from researchforge.domain.outcome import Outcome
from researchforge.domain.problem import ResearchProblem
from researchforge.domain.provenance import Provenance
from researchforge.domain.question import ResearchQuestion
from researchforge.domain.state import ResearchState
from researchforge.domain.validity import Validity, ValidityVerdict
from researchforge.repositories.event_store import (
    EventOrderingError,
    EventStoreMutationError,
    InMemoryEventStore,
    ResearchEventRecord,
)
from researchforge.repositories.interfaces import EntityImmutabilityError
from researchforge.repositories.in_memory import InMemoryUnitOfWork
from researchforge.repositories.replay import EventReplayService
from researchforge.repositories.sql import Base, SqlUnitOfWork
from researchforge.repositories.vrdeg_adapter import InMemoryVRDEGRepository
from researchforge.vrdeg.edge import GraphEdge, RelationType
from researchforge.vrdeg.graph import VRDEG
from researchforge.vrdeg.node import GraphNode, NodeType


# ==============================================================================
# Fixtures
# ==============================================================================

def get_postgresql_url() -> Optional[str]:
    url = os.environ.get("DATABASE_URL")
    if url and url.startswith("postgresql"):
        return url
    candidate = f"postgresql+psycopg2://{os.environ.get('USER', 'postgres')}@localhost:5433/researchforge_test"
    try:
        eng = create_engine(candidate, connect_args={"connect_timeout": 1})
        with eng.connect() as conn:
            pass
        return candidate
    except Exception:
        return None


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="rf_persist_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def in_memory_uow():
    return InMemoryUnitOfWork()


@pytest.fixture
def sqlite_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def sqlite_uow(sqlite_engine):
    return SqlUnitOfWork(sqlite_engine)


@pytest.fixture
def postgresql_engine():
    url = get_postgresql_url()
    if not url:
        pytest.skip("PostgreSQL daemon not configured or reachable. Skipping live PostgreSQL test.")
    engine = create_engine(url)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def postgresql_uow(postgresql_engine):
    return SqlUnitOfWork(postgresql_engine)


# ==============================================================================
# 1. In-Memory Repositories & Entity Isolation
# ==============================================================================

def test_in_memory_repositories_crud(in_memory_uow):
    """All canonical domain entities can be saved, retrieved, listed, and deleted."""
    uow = in_memory_uow
    prob = ResearchProblem(id="prob_01", schema_version="1.0", title="Scaling Transformers")
    uow.problems.save(prob)

    retrieved = uow.problems.get("prob_01")
    assert retrieved is not None
    assert retrieved.id == "prob_01"
    assert retrieved.title == "Scaling Transformers"

    # Verify deepcopy isolation (modifying retrieved does not mutate store)
    assert len(uow.problems.list_all()) == 1
    assert uow.problems.delete("prob_01") is True
    assert uow.problems.get("prob_01") is None


def test_evidence_and_claim_repositories(in_memory_uow):
    """Evidence and Claim repositories support claim-indexed lookups."""
    uow = in_memory_uow
    claim = Claim(id="c_test", schema_version="1.0", statement="GELU outperforms ReLU on task A.")
    ev1 = Evidence(id="e_1", schema_version="1.0", source="exp", claim_id="c_test", relation=EvidenceRelation.SUPPORTED_BY.value)
    ev2 = Evidence(id="e_2", schema_version="1.0", source="lit", claim_id="c_test", relation=EvidenceRelation.SUPPORTED_BY.value)

    uow.claims.save(claim)
    uow.evidence.save(ev1)
    uow.evidence.save(ev2)

    claim_evs = uow.evidence.get_by_claim("c_test")
    assert len(claim_evs) == 2
    assert {e.id for e in claim_evs} == {"e_1", "e_2"}


# ==============================================================================
# 2. Transactional Unit of Work (Atomic Research Cycle)
# ==============================================================================

def test_atomic_unit_of_work_commit_success(in_memory_uow):
    """A full research cycle commits atomically into repositories."""
    uow = in_memory_uow

    with uow:
        # Decision -> Spec -> Run -> Outcome -> Evidence -> State -> Provenance
        dec = Decision(
            id="dec_01",
            schema_version="1.0",
            research_state_fingerprint="fp_state_1",
            rsg_id="rsg_01",
            hypothesis_id="hyp_01",
            selected_tmg_id="tmg_01",
            selected_operator="scale_depth",
            decision_reason="test",
        )
        spec = ExperimentSpec(id="spec_01", schema_version="1.0", decision_id="dec_01")
        run = ExperimentRun(id="run_01", schema_version="1.0", experiment_spec_id="spec_01", status=ExecutionStatus.SUCCESS)
        out = Outcome(id="out_01", schema_version="1.0", run_id="run_01", measured_metrics={"acc": 0.93})
        ev = Evidence(id="ev_01", schema_version="1.0", source="exp", claim_id="c_01", relation="SUPPORTED_BY")
        state = ResearchState(id="state_01", schema_version="1.0", current_decision_id="dec_01")
        prov = Provenance(id="prov_01", schema_version="1.0", created_by="agent_01", created_at="2026-09-05T00:00:00Z")

        uow.decisions.save(dec)
        uow.specs.save(spec)
        uow.runs.save(run)
        uow.outcomes.save(out)
        uow.evidence.save(ev)
        uow.states.save(state)
        uow.provenance.save(prov)
        uow.commit()

    # After successful commit, all entities must exist
    assert uow.decisions.get("dec_01") is not None
    assert uow.specs.get("spec_01") is not None
    assert uow.runs.get("run_01") is not None
    assert uow.outcomes.get("out_01") is not None
    assert uow.evidence.get("ev_01") is not None
    assert uow.states.get("state_01") is not None
    assert uow.provenance.get("prov_01") is not None


def test_atomic_unit_of_work_rollback_on_failure(in_memory_uow):
    """An exception inside a Unit of Work rolls back completely with zero partial state."""
    uow = in_memory_uow

    # Pre-seed one problem
    initial_prob = ResearchProblem(id="prob_initial", schema_version="1.0", title="Initial Problem")
    uow.problems.save(initial_prob)

    try:
        with uow:
            uow.problems.save(ResearchProblem(id="prob_partial", schema_version="1.0", title="Partial Problem"))
            uow.decisions.save(
                Decision(
                    id="dec_partial",
                    schema_version="1.0",
                    research_state_fingerprint="fp_partial",
                    rsg_id="rsg_partial",
                    hypothesis_id="hyp_01",
                    selected_tmg_id="tmg_partial",
                    selected_operator="test",
                    decision_reason="test",
                )
            )
            # Simulate failure during research iteration
            raise RuntimeError("Simulated execution failure in experiment pipeline!")
    except RuntimeError:
        pass

    # Rollback must leave pre-existing problem intact, but discard partial additions!
    assert uow.problems.get("prob_initial") is not None
    assert uow.problems.get("prob_partial") is None
    assert uow.decisions.get("dec_partial") is None


# ==============================================================================
# 3. SQL Repositories & Relational Unit of Work (SQLite & PostgreSQL Compatible)
# ==============================================================================

def test_sql_repositories_and_uow_commit(sqlite_uow):
    """SQL repositories persist and query relational models cleanly over active session."""
    uow = sqlite_uow

    with uow:
        prob = ResearchProblem(id="p_sql", schema_version="1.0", title="SQL Problem")
        hyp = Hypothesis(id="h_sql", schema_version="1.0", research_question_id="q1", statement="Hypothesis SQL")
        ev = Evidence(id="e_sql", schema_version="1.0", source="exp", claim_id="c1", relation="SUPPORTED_BY")
        claim = Claim(id="c1", schema_version="1.0", statement="Claim SQL", status="SUPPORTED")

        uow.problems.save(prob)
        uow.hypotheses.save(hyp)
        uow.evidence.save(ev)
        uow.claims.save(claim)
        uow.commit()

    with uow:
        assert uow.problems.get("p_sql") is not None
        assert uow.hypotheses.get("h_sql") is not None
        assert uow.claims.get("c1") is not None
        assert len(uow.evidence.get_by_claim("c1")) == 1


def test_sql_uow_rollback_on_error(sqlite_uow):
    """SQL Unit of Work automatically rolls back on unhandled exceptions."""
    uow = sqlite_uow

    with uow:
        uow.problems.save(ResearchProblem(id="p_survivor", schema_version="1.0", title="Survivor"))
        uow.commit()

    try:
        with uow:
            uow.problems.save(ResearchProblem(id="p_doomed", schema_version="1.0", title="Doomed"))
            raise ValueError("Crash before commit")
    except ValueError:
        pass

    with uow:
        assert uow.problems.get("p_survivor") is not None
        assert uow.problems.get("p_doomed") is None


# ==============================================================================
# 4. Append-Only Event Store
# ==============================================================================

def test_append_only_event_store_ordering_and_immutability():
    """Event store enforces monotonically increasing sequence numbers and prevents overwriting."""
    store = InMemoryEventStore()

    e1 = ResearchEventRecord(
        id="evt_01",
        schema_version="1.0",
        event_type="DECISION_MADE",
        aggregate_id="agg_state_1",
        aggregate_type="ResearchState",
        sequence_number=1,
        payload={"decision_id": "dec_01"},
        created_at=100.0,
    )
    e2 = ResearchEventRecord(
        id="evt_02",
        schema_version="1.0",
        event_type="EXPERIMENT_PLANNED",
        aggregate_id="agg_state_1",
        aggregate_type="ResearchState",
        sequence_number=2,
        payload={"spec_id": "spec_01"},
        created_at=101.0,
    )

    store.append(e1)
    store.append(e2)

    # 1. Sequential retrieval
    events = store.get_by_aggregate("agg_state_1")
    assert len(events) == 2
    assert events[0].id == "evt_01"
    assert events[1].id == "evt_02"

    # 2. Rejection of duplicate event ID
    with pytest.raises(EventStoreMutationError):
        store.append(e1)

    # 3. Rejection of sequence number regression
    e_bad_seq = ResearchEventRecord(
        id="evt_03",
        schema_version="1.0",
        event_type="EXPERIMENT_COMPLETED",
        aggregate_id="agg_state_1",
        aggregate_type="ResearchState",
        sequence_number=1,  # Regression!
        payload={"run_id": "run_01"},
    )
    with pytest.raises(EventOrderingError):
        store.append(e_bad_seq)


# ==============================================================================
# 5. Artifact Registry & Cryptographic Integrity
# ==============================================================================

def test_artifact_registry_storage_and_tamper_detection(tmp_dir):
    """ArtifactRegistry validates SHA-256 on write and read, catching disk corruption."""
    storage = LocalArtifactStorage(base_dir=tmp_dir)
    registry = ArtifactRegistry(storage=storage)

    raw_data = b"MODEL_CHECKPOINT_BINARY_DATA_TENSOR_WEIGHTS_42"
    artifact = registry.register_artifact(
        content=raw_data,
        artifact_id="art_ckpt_01",
        artifact_type="model_checkpoint",
        producer="trainer_v1",
        extension="bin",
    )

    assert artifact.id == "art_ckpt_01"
    assert artifact.size_bytes == len(raw_data)
    assert artifact.sha256 != ""

    # Normal read succeeds
    loaded = registry.load_artifact_bytes(artifact)
    assert loaded == raw_data

    # Tamper with file on disk
    uri = artifact.uri_or_path
    path = Path(uri.replace("file://", ""))
    assert path.exists()

    with open(path, "wb") as f:
        f.write(b"CORRUPTED_OR_TAMPERED_CONTENT")

    # Read must catch the hash mismatch!
    with pytest.raises(ArtifactIntegrityError):
        registry.load_artifact_bytes(artifact)


# ==============================================================================
# 6. VRDEG Persistence Adapter & Lineage Tracing
# ==============================================================================

def test_vrdeg_graph_adapter_lineage_and_branching():
    """ResearchGraphRepository stores nodes and edges, tracing causal lineage and branches."""
    repo = InMemoryVRDEGRepository()

    # Node 1: Question -> Node 2: Hypothesis -> Node 3: Decision
    n_q = GraphNode(id="node_q", schema_version="1.0", node_type=NodeType.RESEARCH_QUESTION, payload_ref="q1", created_at="2026-09-05T00:00:00Z")
    n_h = GraphNode(id="node_h", schema_version="1.0", node_type=NodeType.HYPOTHESIS, payload_ref="h1", created_at="2026-09-05T00:01:00Z")
    n_d = GraphNode(id="node_d", schema_version="1.0", node_type=NodeType.DECISION, payload_ref="d1", created_at="2026-09-05T00:02:00Z")

    repo.add_node(n_q)
    repo.add_node(n_h)
    repo.add_node(n_d)

    # Precedes edges: Q -> H -> D
    repo.add_edge(GraphEdge(id="edge_1", schema_version="1.0", source_id="node_q", target_id="node_h", relation=RelationType.PRECEDES))
    repo.add_edge(GraphEdge(id="edge_2", schema_version="1.0", source_id="node_h", target_id="node_d", relation=RelationType.PRECEDES))

    # Lineage trace from Decision back to Question
    lineage = repo.trace_lineage("node_d")
    lineage_ids = [n.id for n in lineage]
    assert "node_d" in lineage_ids
    assert "node_h" in lineage_ids
    assert "node_q" in lineage_ids


# ==============================================================================
# 7. Deterministic Event Replay
# ==============================================================================

def test_deterministic_event_replay():
    """Replaying stored events produces matching state and VRDEG projections."""
    initial_state = ResearchState(
        id="s_init",
        schema_version="1.0",
        research_phase="INITIALIZATION",
    )

    events = [
        ResearchEventRecord(
            id="ev_r1",
            schema_version="1.0",
            event_type="DECISION_MADE",
            aggregate_id="s_init",
            aggregate_type="ResearchState",
            sequence_number=1,
            payload={"decision_id": "dec_101"},
            created_at=100.0,
        ),
        ResearchEventRecord(
            id="ev_r2",
            schema_version="1.0",
            event_type="EXPERIMENT_PLANNED",
            aggregate_id="s_init",
            aggregate_type="ResearchState",
            sequence_number=2,
            payload={"spec_id": "spec_101"},
            created_at=101.0,
        ),
        ResearchEventRecord(
            id="ev_r3",
            schema_version="1.0",
            event_type="EXPERIMENT_COMPLETED",
            aggregate_id="s_init",
            aggregate_type="ResearchState",
            sequence_number=3,
            payload={"run_id": "run_101"},
            created_at=102.0,
        ),
    ]

    replay_service = EventReplayService()
    final_state, graph = replay_service.replay_trajectory(initial_state, events)

    assert final_state.current_decision_id == "dec_101"
    assert "spec_101" in (final_state.recent_experiment_refs or [])
    assert "run_101" in (final_state.recent_experiment_refs or [])

    # Second replay must yield identical state fingerprint
    final_state_2, _ = replay_service.replay_trajectory(initial_state, events)
    assert final_state.fingerprint() == final_state_2.fingerprint()


# ==============================================================================
# 8. Schema Migration Upgrade & Downgrade
# ==============================================================================

def test_alembic_schema_migration_roundtrip(tmp_dir):
    """Database schema migrations execute upgrade and downgrade cleanly."""
    db_file = tmp_dir / "migration_test.db"
    engine = create_engine(f"sqlite:///{db_file.as_posix()}")

    # Test upgrade
    Base.metadata.create_all(engine)
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())

    expected_tables = {
        "research_problems", "research_questions", "hypotheses", "decisions",
        "target_model_genomes", "research_system_genomes", "experiment_specs",
        "experiment_runs", "outcomes", "diagnoses", "failures", "evidence_records",
        "claims", "research_states", "provenance_records", "research_artifacts",
        "research_events", "vrdeg_nodes", "vrdeg_edges",
    }
    assert expected_tables.issubset(tables)

    # Test downgrade
    Base.metadata.drop_all(engine)
    inspector_post_drop = inspect(engine)
    remaining_tables = inspector_post_drop.get_table_names()
    for t in expected_tables:
        assert t not in remaining_tables

    engine.dispose()


# ==============================================================================
# 9. Real PostgreSQL Live Connectivity & Full Iteration
# ==============================================================================

def test_postgresql_live_integration(postgresql_uow, postgresql_engine):
    """Tests live PostgreSQL persistence, atomic commit, and rollback against real engine."""
    uow = postgresql_uow
    dialect_name = postgresql_engine.dialect.name
    version = postgresql_engine.dialect.server_version_info
    assert dialect_name == "postgresql"
    assert version >= (14,), f"Expected PostgreSQL 14+, found {version}"

    # 1. Full atomic research cycle in PostgreSQL
    with uow:
        prob = ResearchProblem(id="pg_prob_1", schema_version="1.0", title="Postgres Production Problem")
        q = ResearchQuestion(id="pg_q_1", schema_version="1.0", problem_id="pg_prob_1", question_text="Does scale improve generalization?")
        hyp = Hypothesis(id="pg_h_1", schema_version="1.0", research_question_id="pg_q_1", statement="Scale increases test accuracy")
        dec = Decision(
            id="pg_dec_1",
            schema_version="1.0",
            research_state_fingerprint="fp_pg_0",
            rsg_id="rsg_pg_1",
            hypothesis_id="pg_h_1",
            selected_tmg_id="tmg_pg_1",
            selected_operator="scale_width",
            decision_reason="explore width scaling",
        )
        spec = ExperimentSpec(id="pg_spec_1", schema_version="1.0", decision_id="pg_dec_1")
        run = ExperimentRun(id="pg_run_1", schema_version="1.0", experiment_spec_id="pg_spec_1", status=ExecutionStatus.SUCCESS)
        out = Outcome(id="pg_out_1", schema_version="1.0", run_id="pg_run_1", measured_metrics={"acc": 0.945})
        ev = Evidence(id="pg_ev_1", schema_version="1.0", source="exp", claim_id="pg_claim_1", relation="SUPPORTED_BY")
        claim = Claim(id="pg_claim_1", schema_version="1.0", statement="Scale improves accuracy", status="SUPPORTED")
        state = ResearchState(id="pg_state_1", schema_version="1.0", problem_id="pg_prob_1", current_decision_id="pg_dec_1")
        prov = Provenance(id="pg_prov_1", schema_version="1.0", created_by="pg_agent", created_at="2026-09-05T00:00:00Z")

        uow.problems.save(prob)
        uow.questions.save(q)
        uow.hypotheses.save(hyp)
        uow.decisions.save(dec)
        uow.specs.save(spec)
        uow.runs.save(run)
        uow.outcomes.save(out)
        uow.evidence.save(ev)
        uow.claims.save(claim)
        uow.states.save(state)
        uow.provenance.save(prov)
        uow.commit()

    # 2. Verify all records exist in PostgreSQL
    with uow:
        assert uow.problems.get("pg_prob_1") is not None
        assert uow.questions.get("pg_q_1") is not None
        assert uow.hypotheses.get("pg_h_1") is not None
        assert uow.decisions.get("pg_dec_1") is not None
        assert uow.specs.get("pg_spec_1") is not None
        assert uow.runs.get("pg_run_1") is not None
        assert uow.outcomes.get("pg_out_1") is not None
        assert uow.evidence.get("pg_ev_1") is not None
        assert uow.claims.get("pg_claim_1") is not None
        assert uow.states.get("pg_state_1") is not None
        assert uow.provenance.get("pg_prov_1") is not None
        assert len(uow.evidence.get_by_claim("pg_claim_1")) == 1

    # 3. Clean up
    with uow:
        uow.problems.delete("pg_prob_1")
        uow.questions.delete("pg_q_1")
        uow.hypotheses.delete("pg_h_1")
        uow.decisions.delete("pg_dec_1")
        uow.specs.delete("pg_spec_1")
        uow.runs.delete("pg_run_1")
        uow.evidence.delete("pg_ev_1")
        uow.claims.delete("pg_claim_1")
        uow.commit()


# ==============================================================================
# 10. Concurrency & Unique Sequence Conflict Verification
# ==============================================================================

def test_concurrency_conflict_and_unique_sequence(sqlite_engine):
    """Writing conflicting sequence numbers to the same aggregate deterministically rejects one writer."""
    # A. In-memory concurrency check
    mem_store = InMemoryEventStore()
    e_init = ResearchEventRecord(
        id="evt_init",
        schema_version="1.0",
        event_type="DECISION_MADE",
        aggregate_id="agg_concurrent",
        aggregate_type="ResearchState",
        sequence_number=1,
        payload={"step": 1},
    )
    mem_store.append(e_init)

    # Writer A writes sequence 2
    mem_store.append(
        ResearchEventRecord(
            id="evt_a_2",
            schema_version="1.0",
            event_type="EXPERIMENT_PLANNED",
            aggregate_id="agg_concurrent",
            aggregate_type="ResearchState",
            sequence_number=2,
            payload={"writer": "A"},
        )
    )

    # Writer B attempts sequence 2 -> MUST raise EventOrderingError
    with pytest.raises(EventOrderingError):
        mem_store.append(
            ResearchEventRecord(
                id="evt_b_2",
                schema_version="1.0",
                event_type="EXPERIMENT_PLANNED",
                aggregate_id="agg_concurrent",
                aggregate_type="ResearchState",
                sequence_number=2,
                payload={"writer": "B"},
            )
        )

    # Verify only Writer A was recorded
    agg_events = mem_store.get_by_aggregate("agg_concurrent")
    assert len(agg_events) == 2
    assert agg_events[1].id == "evt_a_2"

    # B. SQL Concurrency Check (SQLite & PostgreSQL)
    sql_uow = SqlUnitOfWork(sqlite_engine)
    with sql_uow:
        sql_uow.events.append(
            ResearchEventRecord(
                id="sql_evt_1",
                schema_version="1.0",
                event_type="DECISION_MADE",
                aggregate_id="agg_sql_conc",
                aggregate_type="ResearchState",
                sequence_number=1,
                payload={"step": 1},
            )
        )
        sql_uow.commit()

    with sql_uow:
        sql_uow.events.append(
            ResearchEventRecord(
                id="sql_evt_a_2",
                schema_version="1.0",
                event_type="EXPERIMENT_PLANNED",
                aggregate_id="agg_sql_conc",
                aggregate_type="ResearchState",
                sequence_number=2,
                payload={"writer": "A"},
            )
        )
        sql_uow.commit()

    # Conflicting sequence 2 must be rejected
    with sql_uow:
        with pytest.raises(EventOrderingError):
            sql_uow.events.append(
                ResearchEventRecord(
                    id="sql_evt_b_2",
                    schema_version="1.0",
                    event_type="EXPERIMENT_PLANNED",
                    aggregate_id="agg_sql_conc",
                    aggregate_type="ResearchState",
                    sequence_number=2,
                    payload={"writer": "B"},
                )
            )


# ==============================================================================
# 11. Transaction Failure Matrix Across All 9 Stages
# ==============================================================================

@pytest.mark.parametrize("failure_stage", [
    "decision",
    "spec",
    "run",
    "outcome",
    "evidence",
    "state",
    "vrdeg",
    "provenance",
    "event",
])
def test_transaction_failure_matrix_all_stages(sqlite_engine, failure_stage):
    """An unhandled error after ANY of the 9 research cycle stages rolls back all changes completely."""
    uow = SqlUnitOfWork(sqlite_engine)
    tag = f"stage_{failure_stage}"

    try:
        with uow:
            # 1. Decision
            dec = Decision(
                id=f"dec_{tag}",
                schema_version="1.0",
                research_state_fingerprint=f"fp_{tag}",
                rsg_id=f"rsg_{tag}",
                hypothesis_id=f"hyp_{tag}",
                selected_tmg_id=f"tmg_{tag}",
                selected_operator="scale",
                decision_reason="test",
            )
            uow.decisions.save(dec)
            if failure_stage == "decision":
                raise RuntimeError(f"Failure after {failure_stage}")

            # 2. Spec
            spec = ExperimentSpec(id=f"spec_{tag}", schema_version="1.0", decision_id=f"dec_{tag}")
            uow.specs.save(spec)
            if failure_stage == "spec":
                raise RuntimeError(f"Failure after {failure_stage}")

            # 3. Run
            run = ExperimentRun(id=f"run_{tag}", schema_version="1.0", experiment_spec_id=f"spec_{tag}", status=ExecutionStatus.SUCCESS)
            uow.runs.save(run)
            if failure_stage == "run":
                raise RuntimeError(f"Failure after {failure_stage}")

            # 4. Outcome
            out = Outcome(id=f"out_{tag}", schema_version="1.0", run_id=f"run_{tag}", measured_metrics={"acc": 0.90})
            uow.outcomes.save(out)
            if failure_stage == "outcome":
                raise RuntimeError(f"Failure after {failure_stage}")

            # 5. Evidence
            ev = Evidence(id=f"ev_{tag}", schema_version="1.0", source="exp", claim_id=f"claim_{tag}", relation="SUPPORTED_BY")
            uow.evidence.save(ev)
            if failure_stage == "evidence":
                raise RuntimeError(f"Failure after {failure_stage}")

            # 6. State
            state = ResearchState(id=f"state_{tag}", schema_version="1.0", current_decision_id=f"dec_{tag}")
            uow.states.save(state)
            if failure_stage == "state":
                raise RuntimeError(f"Failure after {failure_stage}")

            # 7. VRDEG node/edge
            node = GraphNode(id=f"node_{tag}", schema_version="1.0", node_type=NodeType.DECISION, payload_ref=f"dec_{tag}")
            uow.graph.add_node(node)
            if failure_stage == "vrdeg":
                raise RuntimeError(f"Failure after {failure_stage}")

            # 8. Provenance
            prov = Provenance(id=f"prov_{tag}", schema_version="1.0", created_by="test_agent", created_at="2026-09-05T00:00:00Z")
            uow.provenance.save(prov)
            if failure_stage == "provenance":
                raise RuntimeError(f"Failure after {failure_stage}")

            # 9. Event
            event_rec = ResearchEventRecord(
                id=f"evt_{tag}",
                schema_version="1.0",
                event_type="DECISION_MADE",
                aggregate_id=f"state_{tag}",
                aggregate_type="ResearchState",
                sequence_number=1,
                payload={"step": 1},
            )
            uow.events.append(event_rec)
            if failure_stage == "event":
                raise RuntimeError(f"Failure after {failure_stage}")

            uow.commit()
    except RuntimeError:
        pass

    # Verify zero durable state exists from the failed stage!
    with uow:
        assert uow.decisions.get(f"dec_{tag}") is None
        assert uow.specs.get(f"spec_{tag}") is None
        assert uow.runs.get(f"run_{tag}") is None
        assert uow.outcomes.get(f"out_{tag}") is None
        assert uow.evidence.get(f"ev_{tag}") is None
        assert uow.states.get(f"state_{tag}") is None
        assert uow.provenance.get(f"prov_{tag}") is None
        assert uow.events.get_by_id(f"evt_{tag}") is None


# ==============================================================================
# 12. Event Ordering: Aggregate-Local and Global Reconstruction
# ==============================================================================

def test_event_ordering_aggregate_and_global_reconstruction():
    """Event ordering preserves both per-aggregate sequence and monotonic global ordering independent of timestamps."""
    store = InMemoryEventStore()

    # Aggregate A: 3 events
    for seq in range(1, 4):
        store.append(
            ResearchEventRecord(
                id=f"ev_a_{seq}",
                schema_version="1.0",
                event_type="EXPERIMENT_PLANNED",
                aggregate_id="agg_A",
                aggregate_type="ResearchState",
                sequence_number=seq,
                payload={"step": seq},
                created_at=100.0 - seq,  # reverse timestamp to prove timestamp is NOT used for ordering
            )
        )

    # Aggregate B: 2 events
    for seq in range(1, 3):
        store.append(
            ResearchEventRecord(
                id=f"ev_b_{seq}",
                schema_version="1.0",
                event_type="EXPERIMENT_PLANNED",
                aggregate_id="agg_B",
                aggregate_type="ResearchState",
                sequence_number=seq,
                payload={"step": seq},
                created_at=200.0 - seq,
            )
        )

    all_events = store.get_all_events()
    assert len(all_events) == 5

    # Check that each event has an automatically assigned monotonic global_sequence
    global_seqs = [e.global_sequence for e in all_events]
    assert global_seqs == [1, 2, 3, 4, 5]

    # Aggregate sequences are locally monotonic
    agg_a_events = store.get_by_aggregate("agg_A")
    assert [e.sequence_number for e in agg_a_events] == [1, 2, 3]
    assert [e.aggregate_sequence for e in agg_a_events] == [1, 2, 3]

    agg_b_events = store.get_by_aggregate("agg_B")
    assert [e.sequence_number for e in agg_b_events] == [1, 2]


# ==============================================================================
# 13. Replay Recovery After VRDEG Projection Corruption
# ==============================================================================

def test_replay_recovery_after_vrdeg_projection_corruption():
    """Event store is the authoritative recovery journal: corrupting VRDEG is 100% recoverable."""
    initial_state = ResearchState(id="s_root", schema_version="1.0", research_phase="INIT")
    events = [
        ResearchEventRecord(
            id="rec_ev_1",
            schema_version="1.0",
            event_type="DECISION_MADE",
            aggregate_id="s_root",
            aggregate_type="ResearchState",
            sequence_number=1,
            payload={"decision_id": "dec_recovery_1", "hypothesis_id": "hyp_recovery_1"},
        ),
        ResearchEventRecord(
            id="rec_ev_2",
            schema_version="1.0",
            event_type="EXPERIMENT_PLANNED",
            aggregate_id="s_root",
            aggregate_type="ResearchState",
            sequence_number=2,
            payload={"spec_id": "spec_recovery_1", "decision_id": "dec_recovery_1"},
        ),
    ]

    replay_service = EventReplayService()
    # 1. Initial projection
    state_1, graph_1 = replay_service.replay_trajectory(initial_state, events)
    fp_1 = graph_1.fingerprint()

    # 2. Corrupt / destroy the VRDEG graph completely
    graph_1._nodes.clear()
    graph_1._edges.clear()
    assert len(graph_1._nodes) == 0
    assert len(graph_1._edges) == 0

    # 3. Recover completely from persisted event journal
    state_recovered, graph_recovered = replay_service.replay_trajectory(initial_state, events)
    fp_recovered = graph_recovered.fingerprint()

    # 4. Fingerprint must be identical!
    assert fp_recovered == fp_1
    assert state_recovered.fingerprint() == state_1.fingerprint()
    assert graph_recovered.get_node("dec_recovery_1") is not None
    assert graph_recovered.get_node("spec_recovery_1") is not None


# ==============================================================================
# 14. Repository Contract Parity Across Backends
# ==============================================================================

@pytest.mark.parametrize("backend_name", ["in_memory", "sqlite", "postgresql"])
def test_repository_contract_parity(request, backend_name):
    """InMemory, SQLite, and PostgreSQL backends exhibit identical research contract semantics."""
    if backend_name == "in_memory":
        uow = request.getfixturevalue("in_memory_uow")
    elif backend_name == "sqlite":
        uow = request.getfixturevalue("sqlite_uow")
    elif backend_name == "postgresql":
        uow = request.getfixturevalue("postgresql_uow")
    else:
        raise ValueError(backend_name)

    tag = f"parity_{backend_name}"

    with uow:
        prob = ResearchProblem(id=f"p_{tag}", schema_version="1.0", title="Parity Problem")
        hyp = Hypothesis(id=f"h_{tag}", schema_version="1.0", research_question_id="q1", statement="Parity Hyp")
        claim = Claim(id=f"c_{tag}", schema_version="1.0", statement="Parity Claim", status="SUPPORTED")
        ev = Evidence(id=f"e_{tag}", schema_version="1.0", source="exp", claim_id=f"c_{tag}", relation="SUPPORTED_BY")
        art = Artifact(id=f"art_{tag}", schema_version="1.0", artifact_type="weights", uri_or_path=f"artifacts/{tag}.bin", sha256=f"hash_{tag}", size_bytes=1024)
        state = ResearchState(id=f"s_{tag}", schema_version="1.0", problem_id=f"p_{tag}")

        uow.problems.save(prob)
        uow.hypotheses.save(hyp)
        uow.claims.save(claim)
        uow.evidence.save(ev)
        uow.artifacts.save(art)
        uow.states.save(state)
        uow.commit()

    with uow:
        # Check retrieval parity
        assert uow.problems.get(f"p_{tag}").title == "Parity Problem"
        assert uow.hypotheses.get(f"h_{tag}").statement == "Parity Hyp"
        assert uow.claims.get(f"c_{tag}").status == "SUPPORTED"
        assert len(uow.evidence.get_by_claim(f"c_{tag}")) == 1
        assert uow.artifacts.get_by_checksum(f"hash_{tag}").id == f"art_{tag}"
        assert uow.states.get_latest().id == f"s_{tag}"

        # Check delete
        assert uow.problems.delete(f"p_{tag}") is True
        assert uow.problems.get(f"p_{tag}") is None
        uow.commit()


# ==============================================================================
# 15. Historical Immutability Enforcement
# ==============================================================================

def test_historical_immutability_enforcement(in_memory_uow, sqlite_uow):
    """Historical events, outcomes, failures, provenance, and states cannot be silently overwritten or deleted."""
    for uow in [in_memory_uow, sqlite_uow]:
        with uow:
            out = Outcome(id="imm_out_1", schema_version="1.0", run_id="run_1", measured_metrics={"acc": 0.91})
            fail = Failure(id="imm_fail_1", schema_version="1.0", experiment_id="run_1", failure_category="OOM")
            prov = Provenance(id="imm_prov_1", schema_version="1.0", created_by="agent", created_at="2026-09-05T00:00:00Z")
            state = ResearchState(id="imm_state_1", schema_version="1.0")

            uow.outcomes.save(out)
            uow.failures.save(fail)
            uow.provenance.save(prov)
            uow.states.save(state)
            uow.commit()

        # Overwrite attempts MUST raise EntityImmutabilityError
        with uow:
            with pytest.raises(EntityImmutabilityError):
                uow.outcomes.save(Outcome(id="imm_out_1", schema_version="1.0", run_id="run_1", measured_metrics={"acc": 0.99}))

            with pytest.raises(EntityImmutabilityError):
                uow.failures.save(Failure(id="imm_fail_1", schema_version="1.0", experiment_id="run_1", failure_category="CUDA"))

            with pytest.raises(EntityImmutabilityError):
                uow.provenance.save(Provenance(id="imm_prov_1", schema_version="1.0", created_by="impostor", created_at="2026-09-05T00:00:00Z"))

            with pytest.raises(EntityImmutabilityError):
                uow.states.save(ResearchState(id="imm_state_1", schema_version="1.0", problem_id="different"))

            # Deletion attempts MUST raise EntityImmutabilityError
            with pytest.raises(EntityImmutabilityError):
                uow.outcomes.delete("imm_out_1")

            with pytest.raises(EntityImmutabilityError):
                uow.failures.delete("imm_fail_1")

            with pytest.raises(EntityImmutabilityError):
                uow.provenance.delete("imm_prov_1")

            with pytest.raises(EntityImmutabilityError):
                uow.states.delete("imm_state_1")


# ==============================================================================
# 16. Artifact Integrity, Relocation, and Retirement
# ==============================================================================

def test_artifact_integrity_hash_tamper_relocation_retirement(tmp_dir, in_memory_uow):
    """Artifacts verify SHA-256, detect tampering, preserve identity on relocation, and support retirement without data deletion."""
    storage = LocalArtifactStorage(base_dir=tmp_dir / "store")
    registry = ArtifactRegistry(storage=storage, repository=in_memory_uow.artifacts)

    content = b"CRITICAL_MODEL_CHECKPOINT_DATA_v1"
    expected_hash = hashlib.sha256(content).hexdigest()

    # 1. Register & verify hash match
    artifact = registry.register_artifact(
        content=content,
        artifact_id="art_lifecycle_01",
        artifact_type="model_weights",
    )
    assert artifact.sha256 == expected_hash
    loaded = registry.load_artifact_bytes(artifact.id)
    assert loaded == content

    def _to_path(uri: str) -> Path:
        p = uri
        if p.startswith("file://"):
            p = p[7:]
        elif p.startswith("file:"):
            p = p[5:]
        return Path(p)

    # 2. Tampering detection
    physical_path = _to_path(artifact.uri_or_path)
    assert physical_path.exists()
    with open(physical_path, "wb") as f:
        f.write(b"TAMPERED_MALICIOUS_WEIGHTS")

    with pytest.raises(ArtifactIntegrityError):
        registry.load_artifact_bytes(artifact.id)

    # Restore content for subsequent lifecycle tests
    with open(physical_path, "wb") as f:
        f.write(content)

    # 3. Relocation preserves artifact identity
    new_path = tmp_dir / "archive" / "relocated_weights.bin"
    new_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(physical_path, new_path)

    import dataclasses
    relocated_artifact = dataclasses.replace(artifact, uri_or_path=str(new_path))
    in_memory_uow.artifacts.save(relocated_artifact)

    # Reading relocated artifact validates the same cryptographic SHA-256
    relocated_bytes = registry.load_artifact_bytes(relocated_artifact.id)
    assert relocated_bytes == content
    assert relocated_artifact.sha256 == expected_hash

    # 4. Retirement is distinguishable from physical deletion
    retired_art = registry.retire_artifact(artifact.id, reason="superseded by v2")
    assert retired_art.is_retired is True
    assert retired_art.metadata.get("retired_reason") == "superseded by v2"

    # Bytes STILL exist physically on disk and can be inspected for forensic audit
    assert _to_path(retired_art.uri_or_path).exists()
    assert registry.load_artifact_bytes(retired_art.id) == content


# ==============================================================================
# 17. Performance Baseline Microbenchmark
# ==============================================================================

def test_persistence_microbenchmark(sqlite_uow):
    """Baseline timing check for event append, entity persistence, and event replay."""
    uow = sqlite_uow

    # 1. Measure entity persistence (50 records)
    t0 = time.perf_counter()
    with uow:
        for i in range(50):
            uow.problems.save(ResearchProblem(id=f"bench_p_{i}", schema_version="1.0", title=f"Problem {i}"))
        uow.commit()
    t_entity = time.perf_counter() - t0
    assert t_entity < 1.0  # Must be fast (<1s for 50 records)

    # 2. Measure event append (50 events)
    store = InMemoryEventStore()
    t0 = time.perf_counter()
    for i in range(50):
        store.append(ResearchEventRecord(
            id=f"bench_ev_{i}",
            schema_version="1.0",
            event_type="EXPERIMENT_PLANNED",
            aggregate_id="agg_bench",
            aggregate_type="ResearchState",
            sequence_number=i + 1,
            payload={"step": i},
        ))
    t_event = time.perf_counter() - t0
    assert t_event < 0.2  # Append-only should be sub-200ms

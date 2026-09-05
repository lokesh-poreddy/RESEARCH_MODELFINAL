"""researchforge/benchmarks/execution/preflight.py — 8-Point Preflight Verification Gate.

Phase 12B.2 Core Invariants:
1. Gate 1: Cohort Specification Integrity (v1.0.1, fingerprint 591e1a0bf62d9f0e9bfe7d717b66582dd89b1c7cc65d45082ff26607bc031e66)
2. Gate 2: Statistical Analysis Plan (SAP) Integrity (v1.0.0, confirmatory H1, secondary H2-H4)
3. Gate 3: Task Materialization Parity (bitwise-identical array hashes across repeated generations)
4. Gate 4: Transfer Relation Verification (two-tier regimes, sequence vs effective, zero prior on cold start)
5. Gate 5: Future-Information Leakage Prevention (causal barrier, zero forward visibility)
6. Gate 6: State Isolation & Persistence Semantics (fresh memory on cold start, inheritance on continuous)
7. Gate 7: Controller Micro-Execution Test (1 gen x 2 candidates end-to-end telemetry check)
8. Gate 8: Manifest Completeness & DAG Integrity (540 entries, 32,400 trials, zero DAG cycles)
"""
from __future__ import annotations

import hashlib
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ...domain.base import _canonical_json
from ...pipeline.controller import ResearchController
from ..cohort import (
    BenchmarkCondition,
    TaskFamily,
    TransferRegime,
    create_canonical_cohort_spec,
)
from ..continuity.evaluator import ContinuityEvaluator
from ..continuity.models import ExperiencePartition, FutureInformationLeakageError
from ..statistical.plan import HypothesisType, create_canonical_sap
from .manifest import generate_execution_manifest
from .materialization import (
    compute_canonical_task_content_hash,
    materialize_benchmark_task,
)
from .models import (
    ExecutionManifest,
    PreflightAttestation,
    PreflightCheckResult,
    PreflightGateError,
    PreflightValidationReport,
    RunManifestEntry,
    get_software_commit_info,
)


EXPECTED_COHORT_FINGERPRINT = "591e1a0bf62d9f0e9bfe7d717b66582dd89b1c7cc65d45082ff26607bc031e66"
EXPECTED_COHORT_VERSION = "1.0.1"
EXPECTED_SAP_VERSION = "1.0.0"


class PreflightValidator:
    """Executes the 8-point preflight verification gate suite before any benchmark execution."""

    def __init__(
        self,
        cohort_spec=None,
        sap_spec=None,
    ) -> None:
        self.cohort_spec = cohort_spec or create_canonical_cohort_spec()
        self.sap_spec = sap_spec or create_canonical_sap()
        self.evaluator = ContinuityEvaluator()

    # ── Gate 1: Cohort Specification Integrity ──────────────────────────────────
    def check_gate1_cohort_integrity(self) -> PreflightCheckResult:
        t0 = time.time()
        try:
            cohort = self.cohort_spec
            errors = []
            if cohort.specification_version != EXPECTED_COHORT_VERSION:
                errors.append(f"Cohort version mismatch: expected {EXPECTED_COHORT_VERSION}, got {cohort.specification_version}")
            if cohort.cohort_fingerprint != EXPECTED_COHORT_FINGERPRINT:
                errors.append(f"Cohort fingerprint mismatch: expected {EXPECTED_COHORT_FINGERPRINT}, got {cohort.cohort_fingerprint}")
            if len(cohort.tasks) != 6:
                errors.append(f"Expected 6 tasks in universe, got {len(cohort.tasks)}")
            if len(cohort.conditions) != 6:
                errors.append(f"Expected 6 conditions, got {len(cohort.conditions)}")
            if len(cohort.orderings) != 3:
                errors.append(f"Expected 3 orderings, got {len(cohort.orderings)}")
            if len(cohort.seeds) != 5:
                errors.append(f"Expected 5 seeds, got {len(cohort.seeds)}")

            latency = (time.time() - t0) * 1000.0
            if errors:
                return PreflightCheckResult(
                    gate_number=1,
                    gate_name="Cohort Specification Integrity",
                    status="FAILED",
                    details="; ".join(errors),
                    evidence={"cohort_fingerprint": cohort.cohort_fingerprint, "errors": errors},
                    latency_ms=latency,
                )

            return PreflightCheckResult(
                gate_number=1,
                gate_name="Cohort Specification Integrity",
                status="PASSED",
                details="Cohort specification v1.0.1 verified against sealed cryptographic fingerprint.",
                evidence={
                    "cohort_version": cohort.specification_version,
                    "cohort_fingerprint": cohort.cohort_fingerprint,
                    "parent_fingerprint": cohort.parent_fingerprint,
                    "task_count": len(cohort.tasks),
                    "condition_count": len(cohort.conditions),
                    "ordering_count": len(cohort.orderings),
                    "seed_count": len(cohort.seeds),
                },
                latency_ms=latency,
            )
        except Exception as e:
            return PreflightCheckResult(
                gate_number=1,
                gate_name="Cohort Specification Integrity",
                status="FAILED",
                details=f"Exception during Gate 1 check: {e}",
                evidence={"exception": str(e)},
                latency_ms=(time.time() - t0) * 1000.0,
            )

    # ── Gate 2: SAP Specification Integrity ────────────────────────────────────
    def check_gate2_sap_integrity(self) -> PreflightCheckResult:
        t0 = time.time()
        try:
            sap = self.sap_spec
            sap_fp = getattr(sap, "plan_fingerprint", getattr(sap, "sap_fingerprint", ""))
            errors = []
            if sap.plan_version != EXPECTED_SAP_VERSION:
                errors.append(f"SAP version mismatch: expected {EXPECTED_SAP_VERSION}, got {sap.plan_version}")
            if not sap_fp:
                errors.append("SAP fingerprint is empty or not sealed")
            if len(sap.hypotheses) != 4:
                errors.append(f"Expected 4 hypotheses in SAP, got {len(sap.hypotheses)}")

            # Verify H1 primary confirmatory contract
            h1 = sap.hypotheses.get("H1")
            if not h1:
                errors.append("H1 missing from SAP")
            else:
                if h1.hypothesis_type != HypothesisType.PRIMARY_CONFIRMATORY:
                    errors.append(f"H1 must be PRIMARY_CONFIRMATORY, got {h1.hypothesis_type}")
                if h1.endpoint != "decision_quality":
                    errors.append(f"H1 endpoint must be decision_quality, got {h1.endpoint}")
                if h1.test_condition != "CONTINUOUS_EXPERIENCE":
                    errors.append(f"H1 test_condition must be CONTINUOUS_EXPERIENCE, got {h1.test_condition}")
                if h1.control_condition != "COLD_START":
                    errors.append(f"H1 control_condition must be COLD_START, got {h1.control_condition}")
                if h1.regime_filter != "SAME_FAMILY":
                    errors.append(f"H1 regime_filter must be SAME_FAMILY, got {h1.regime_filter}")

            # Verify H2-H4 secondary family
            for hid in ("H2", "H3", "H4"):
                h = sap.hypotheses.get(hid)
                if not h:
                    errors.append(f"{hid} missing from SAP")
                elif h.hypothesis_type != HypothesisType.SECONDARY_EXPLORATORY:
                    errors.append(f"{hid} must be SECONDARY_EXPLORATORY, got {h.hypothesis_type}")

            h4 = sap.hypotheses.get("H4")
            if h4 and h4.expected_direction != "NON_MONOTONIC":
                errors.append(f"H4 expected_direction must be NON_MONOTONIC, got {h4.expected_direction}")

            latency = (time.time() - t0) * 1000.0
            if errors:
                return PreflightCheckResult(
                    gate_number=2,
                    gate_name="SAP Specification Integrity",
                    status="FAILED",
                    details="; ".join(errors),
                    evidence={"sap_fingerprint": sap_fp, "errors": errors},
                    latency_ms=latency,
                )

            return PreflightCheckResult(
                gate_number=2,
                gate_name="SAP Specification Integrity",
                status="PASSED",
                details="SAP v1.0.0 verified: H1 primary confirmatory endpoint, H2-H4 secondary multiplicity family, H4 non-monotonic contrast.",
                evidence={
                    "sap_version": sap.plan_version,
                    "sap_fingerprint": sap_fp,
                    "hypothesis_count": len(sap.hypotheses),
                    "primary_hypothesis": "H1",
                    "alpha_primary": 0.05,
                },
                latency_ms=latency,
            )
        except Exception as e:
            return PreflightCheckResult(
                gate_number=2,
                gate_name="SAP Specification Integrity",
                status="FAILED",
                details=f"Exception during Gate 2 check: {e}",
                evidence={"exception": str(e)},
                latency_ms=(time.time() - t0) * 1000.0,
            )

    # ── Gate 3: Task Materialization Parity ─────────────────────────────────────
    def check_gate3_materialization_parity(self) -> PreflightCheckResult:
        t0 = time.time()
        try:
            task_hashes: Dict[str, str] = {}
            mismatches: List[str] = []

            for task_id in self.cohort_spec.tasks.keys():
                m1 = materialize_benchmark_task(task_id, seed=0)
                m2 = materialize_benchmark_task(task_id, seed=0)

                # Array equality checks
                if not np.array_equal(m1.task.X_train, m2.task.X_train):
                    mismatches.append(f"{task_id}: X_train bitwise mismatch")
                if not np.array_equal(m1.task.y_train, m2.task.y_train):
                    mismatches.append(f"{task_id}: y_train bitwise mismatch")
                if not np.array_equal(m1.task.X_val, m2.task.X_val):
                    mismatches.append(f"{task_id}: X_val bitwise mismatch")
                if not np.array_equal(m1.task.y_val, m2.task.y_val):
                    mismatches.append(f"{task_id}: y_val bitwise mismatch")

                h1 = compute_canonical_task_content_hash(m1.task)
                h2 = compute_canonical_task_content_hash(m2.task)
                if h1 != h2:
                    mismatches.append(f"{task_id}: canonical content hash mismatch {h1} != {h2}")

                task_hashes[task_id] = h1

            latency = (time.time() - t0) * 1000.0
            if mismatches:
                return PreflightCheckResult(
                    gate_number=3,
                    gate_name="Task Materialization Parity",
                    status="FAILED",
                    details="; ".join(mismatches),
                    evidence={"mismatches": mismatches},
                    latency_ms=latency,
                )

            return PreflightCheckResult(
                gate_number=3,
                gate_name="Task Materialization Parity",
                status="PASSED",
                details="Bitwise array equality and canonical byte-level hash parity verified across all 6 benchmark tasks.",
                evidence={"task_content_hashes": task_hashes},
                latency_ms=latency,
            )
        except Exception as e:
            return PreflightCheckResult(
                gate_number=3,
                gate_name="Task Materialization Parity",
                status="FAILED",
                details=f"Exception during Gate 3 check: {e}",
                evidence={"exception": str(e)},
                latency_ms=(time.time() - t0) * 1000.0,
            )

    # ── Gate 4: Transfer Relation Verification ──────────────────────────────────
    def check_gate4_transfer_relations(self, manifest: ExecutionManifest) -> PreflightCheckResult:
        t0 = time.time()
        try:
            violations = []
            for entry in manifest.entries:
                # Invariant 1: Position 0 must ALWAYS have NO_PRIOR_EXPERIENCE
                if entry.sequence_position == 0:
                    if entry.sequence_transfer_regime != TransferRegime.NO_PRIOR_EXPERIENCE:
                        violations.append(
                            f"{entry.entry_id}: pos 0 sequence_transfer_regime is {entry.sequence_transfer_regime.value}, must be NO_PRIOR_EXPERIENCE"
                        )
                    if entry.condition_effective_regime != TransferRegime.NO_PRIOR_EXPERIENCE:
                        violations.append(
                            f"{entry.entry_id}: pos 0 condition_effective_regime is {entry.condition_effective_regime.value}, must be NO_PRIOR_EXPERIENCE"
                        )

                # Invariant 2: COLD_START and NO_MEMORY must ALWAYS have condition_effective_regime == NO_PRIOR_EXPERIENCE
                if entry.condition in (BenchmarkCondition.COLD_START, BenchmarkCondition.NO_MEMORY):
                    if entry.condition_effective_regime != TransferRegime.NO_PRIOR_EXPERIENCE:
                        violations.append(
                            f"{entry.entry_id}: {entry.condition.value} condition_effective_regime is {entry.condition_effective_regime.value}, must be NO_PRIOR_EXPERIENCE"
                        )

                # Invariant 3: Continuous conditions must inherit sequence_transfer_regime
                elif entry.sequence_position > 0:
                    if entry.condition_effective_regime != entry.sequence_transfer_regime:
                        violations.append(
                            f"{entry.entry_id}: {entry.condition.value} pos {entry.sequence_position} effective regime {entry.condition_effective_regime.value} != sequence regime {entry.sequence_transfer_regime.value}"
                        )

            # Specific ordering checks
            forward_pos1 = [e for e in manifest.entries if e.ordering == "forward" and e.sequence_position == 1][0]
            if forward_pos1.sequence_transfer_regime != TransferRegime.SAME_FAMILY:
                violations.append(f"forward pos 1 sequence regime should be SAME_FAMILY, got {forward_pos1.sequence_transfer_regime.value}")

            forward_pos2 = [e for e in manifest.entries if e.ordering == "forward" and e.sequence_position == 2][0]
            if forward_pos2.sequence_transfer_regime != TransferRegime.SAME_FAMILY:
                violations.append(f"forward pos 2 sequence regime should be SAME_FAMILY, got {forward_pos2.sequence_transfer_regime.value}")

            forward_pos3 = [e for e in manifest.entries if e.ordering == "forward" and e.sequence_position == 3][0]
            if forward_pos3.sequence_transfer_regime != TransferRegime.CROSS_FAMILY:
                violations.append(f"forward pos 3 sequence regime should be CROSS_FAMILY, got {forward_pos3.sequence_transfer_regime.value}")

            reverse_pos1 = [e for e in manifest.entries if e.ordering == "reverse" and e.sequence_position == 1][0]
            if reverse_pos1.sequence_transfer_regime != TransferRegime.UNRELATED:
                violations.append(f"reverse pos 1 sequence regime should be UNRELATED, got {reverse_pos1.sequence_transfer_regime.value}")

            cross_pos1 = [e for e in manifest.entries if e.ordering == "cross_first" and e.sequence_position == 1][0]
            if cross_pos1.sequence_transfer_regime != TransferRegime.CROSS_FAMILY:
                violations.append(f"cross_first pos 1 sequence regime should be CROSS_FAMILY, got {cross_pos1.sequence_transfer_regime.value}")

            latency = (time.time() - t0) * 1000.0
            if violations:
                return PreflightCheckResult(
                    gate_number=4,
                    gate_name="Transfer Relation Verification",
                    status="FAILED",
                    details=f"Found {len(violations)} transfer regime violations: {'; '.join(violations[:5])}",
                    evidence={"violations_count": len(violations), "first_violations": violations[:5]},
                    latency_ms=latency,
                )

            return PreflightCheckResult(
                gate_number=4,
                gate_name="Transfer Relation Verification",
                status="PASSED",
                details="Two-tier transfer regimes verified across all 540 entries: position 0 is strictly NO_PRIOR_EXPERIENCE; COLD_START/NO_MEMORY effectively zero; continuous conditions preserve sequence relations.",
                evidence={
                    "total_entries_verified": len(manifest.entries),
                    "forward_pos1_regime": forward_pos1.sequence_transfer_regime.value,
                    "reverse_pos1_regime": reverse_pos1.sequence_transfer_regime.value,
                    "cross_first_pos1_regime": cross_pos1.sequence_transfer_regime.value,
                },
                latency_ms=latency,
            )
        except Exception as e:
            return PreflightCheckResult(
                gate_number=4,
                gate_name="Transfer Relation Verification",
                status="FAILED",
                details=f"Exception during Gate 4 check: {e}",
                evidence={"exception": str(e)},
                latency_ms=(time.time() - t0) * 1000.0,
            )

    # ── Gate 5: Future-Information Leakage Prevention ───────────────────────────
    def check_gate5_future_leakage_prevention(self, manifest: ExecutionManifest) -> PreflightCheckResult:
        t0 = time.time()
        try:
            leakage_errors = []
            # Group entries by sequence_group_id
            groups: Dict[str, List[RunManifestEntry]] = defaultdict(list)
            for e in manifest.entries:
                groups[e.sequence_group_id].append(e)

            for group_id, entries in groups.items():
                entries.sort(key=lambda x: x.sequence_position)
                all_group_task_ids = [e.task_id for e in entries]

                for idx, entry in enumerate(entries):
                    expected_priors = tuple(all_group_task_ids[:idx])
                    future_task_ids = set(all_group_task_ids[idx + 1:])

                    # 1. Check entry.prior_task_ids matches exact causal history
                    if entry.prior_task_ids != expected_priors:
                        leakage_errors.append(
                            f"{entry.entry_id}: prior_task_ids {entry.prior_task_ids} != expected {expected_priors}"
                        )

                    # 2. Check no future task is present in priors
                    overlap = set(entry.prior_task_ids).intersection(future_task_ids)
                    if overlap:
                        leakage_errors.append(
                            f"{entry.entry_id}: future task IDs {overlap} leaked into prior_task_ids"
                        )

                    # 3. Check partition sealing contract raises FutureInformationLeakageError on future query
                    partition = ExperiencePartition.create(
                        task_id=entry.task_id,
                        task_sequence_index=idx,
                        allowed_task_ids=list(expected_priors),
                        prior_memory_fingerprint="test_fp",
                        current_task_fingerprint=entry.task_fingerprint,
                        future_task_ids=list(future_task_ids),
                    )
                    for f_id in future_task_ids:
                        try:
                            partition.verify_access(f_id)
                            leakage_errors.append(f"{entry.entry_id}: partition failed to block future access to {f_id}")
                        except FutureInformationLeakageError:
                            pass  # Expected behavior

            latency = (time.time() - t0) * 1000.0
            if leakage_errors:
                return PreflightCheckResult(
                    gate_number=5,
                    gate_name="Future-Information Leakage Prevention",
                    status="FAILED",
                    details=f"{len(leakage_errors)} causal barrier violations detected: {'; '.join(leakage_errors[:5])}",
                    evidence={"leakage_count": len(leakage_errors), "first_errors": leakage_errors[:5]},
                    latency_ms=latency,
                )

            return PreflightCheckResult(
                gate_number=5,
                gate_name="Future-Information Leakage Prevention",
                status="PASSED",
                details="Causal ordering verified across all 90 sequence groups: zero forward information leakage, strict topological dependency enforcement, partition sealing access barriers intact.",
                evidence={
                    "sequence_groups_checked": len(groups),
                    "entries_checked": len(manifest.entries),
                    "access_block_verified": True,
                },
                latency_ms=latency,
            )
        except Exception as e:
            return PreflightCheckResult(
                gate_number=5,
                gate_name="Future-Information Leakage Prevention",
                status="FAILED",
                details=f"Exception during Gate 5 check: {e}",
                evidence={"exception": str(e)},
                latency_ms=(time.time() - t0) * 1000.0,
            )

    # ── Gate 6: State Isolation & Persistence Semantics ─────────────────────────
    def check_gate6_state_isolation_and_persistence(self) -> PreflightCheckResult:
        t0 = time.time()
        try:
            isolation_failures = []

            # 1. COLD_START test: Fresh state for each task
            m_task1 = materialize_benchmark_task("digits_0_4", seed=0)
            m_task2 = materialize_benchmark_task("digits_5_9", seed=0)

            ctrl_cs1 = ResearchController(m_task1.task, condition="cold_start", seed=0)
            initial_cold_fp = ctrl_cs1.export_memory_fingerprint()
            ctrl_cs1.run(n_generations=1)
            after_cs1_fp = ctrl_cs1.export_memory_fingerprint()

            ctrl_cs2 = ResearchController(m_task2.task, condition="cold_start", seed=0)
            before_cs2_fp = ctrl_cs2.export_memory_fingerprint()

            if before_cs2_fp != initial_cold_fp:
                isolation_failures.append(f"COLD_START state not fresh on task 2: {before_cs2_fp} != {initial_cold_fp}")
            if before_cs2_fp == after_cs1_fp:
                isolation_failures.append("COLD_START task 2 inherited task 1 state")

            # 2. NO_MEMORY test: Memory disabled
            ctrl_nm = ResearchController(m_task1.task, condition="no_memory", seed=0)
            nm_fp = ctrl_nm.export_memory_fingerprint()
            if nm_fp != "no_memory_disabled":
                isolation_failures.append(f"NO_MEMORY did not produce 'no_memory_disabled' fingerprint: got {nm_fp}")
            if ctrl_nm.use_memory is not False:
                isolation_failures.append("NO_MEMORY ctrl.use_memory is not False")

            # 3. CONTINUOUS_EXPERIENCE test: Experience successfully inherited
            ctrl_ce1 = ResearchController(m_task1.task, condition="continuous_experience", seed=0)
            ctrl_ce1.run(n_generations=1)
            accumulated_ce1_fp = ctrl_ce1.export_memory_fingerprint()

            ctrl_ce2 = ResearchController(
                m_task2.task,
                condition="continuous_experience",
                seed=0,
                ecrm=ctrl_ce1.ecrm,
                trajectory_memory=ctrl_ce1.trajectory_memory,
                adaptive_trajectory_memory=ctrl_ce1.adaptive_trajectory_memory,
                policy_learner=ctrl_ce1.policy_learner,
                failed_signatures=ctrl_ce1._failed_signatures,
            )
            inherited_ce2_fp = ctrl_ce2.export_memory_fingerprint()
            if inherited_ce2_fp != accumulated_ce1_fp:
                isolation_failures.append(
                    f"CONTINUOUS_EXPERIENCE failed to inherit memory: {inherited_ce2_fp} != {accumulated_ce1_fp}"
                )

            latency = (time.time() - t0) * 1000.0
            if isolation_failures:
                return PreflightCheckResult(
                    gate_number=6,
                    gate_name="State Isolation & Persistence Semantics",
                    status="FAILED",
                    details="; ".join(isolation_failures),
                    evidence={"failures": isolation_failures},
                    latency_ms=latency,
                )

            return PreflightCheckResult(
                gate_number=6,
                gate_name="State Isolation & Persistence Semantics",
                status="PASSED",
                details="Verified state semantics: COLD_START enforces fresh empty state per task; NO_MEMORY disables memory; CONTINUOUS_EXPERIENCE preserves accumulated state across tasks.",
                evidence={
                    "cold_start_fresh_verified": True,
                    "no_memory_disabled_verified": True,
                    "continuous_inheritance_verified": True,
                },
                latency_ms=latency,
            )
        except Exception as e:
            return PreflightCheckResult(
                gate_number=6,
                gate_name="State Isolation & Persistence Semantics",
                status="FAILED",
                details=f"Exception during Gate 6 check: {e}",
                evidence={"exception": str(e)},
                latency_ms=(time.time() - t0) * 1000.0,
            )

    # ── Gate 7: Controller Micro-Execution Test ─────────────────────────────────
    def check_gate7_controller_micro_execution(self) -> PreflightCheckResult:
        t0 = time.time()
        try:
            m_task = materialize_benchmark_task("digits_0_4", seed=0)
            ctrl = ResearchController(
                m_task.task,
                condition="cold_start",
                seed=0,
                population_size=2,
            )
            result = ctrl.run(n_generations=1)
            dq = self.evaluator.calculate_decision_quality(result.trials, set())

            checks = []
            if result.best_metric <= 0.0:
                checks.append(f"Invalid best_metric: {result.best_metric}")
            if len(result.trials) == 0:
                checks.append("Zero trials executed in micro-run")
            if not (0.0 <= dq <= 1.0):
                checks.append(f"DQ out of bounds [0, 1]: {dq}")

            latency = (time.time() - t0) * 1000.0
            if checks:
                return PreflightCheckResult(
                    gate_number=7,
                    gate_name="Controller Micro-Execution Test",
                    status="FAILED",
                    details="; ".join(checks),
                    evidence={"checks": checks},
                    latency_ms=latency,
                )

            return PreflightCheckResult(
                gate_number=7,
                gate_name="Controller Micro-Execution Test",
                status="PASSED",
                details="End-to-end micro-execution (1 gen x 2 candidates) succeeded: valid metric optimization, trial telemetry logging, and bounded DQ score calculated.",
                evidence={
                    "trials_executed": len(result.trials),
                    "best_metric": result.best_metric,
                    "decision_quality": dq,
                    "task": "digits_0_4",
                },
                latency_ms=latency,
            )
        except Exception as e:
            return PreflightCheckResult(
                gate_number=7,
                gate_name="Controller Micro-Execution Test",
                status="FAILED",
                details=f"Exception during Gate 7 check: {e}",
                evidence={"exception": str(e)},
                latency_ms=(time.time() - t0) * 1000.0,
            )

    # ── Gate 8: Manifest Completeness & DAG Integrity ───────────────────────────
    def check_gate8_manifest_completeness_and_dag(self, manifest: ExecutionManifest) -> PreflightCheckResult:
        t0 = time.time()
        try:
            errors = []
            if manifest.entry_count != 540:
                errors.append(f"Manifest entry_count must be 540, got {manifest.entry_count}")
            if len(manifest.entries) != 540:
                errors.append(f"Manifest entries length must be 540, got {len(manifest.entries)}")
            if manifest.sequence_group_count != 90:
                errors.append(f"Manifest sequence_group_count must be 90, got {manifest.sequence_group_count}")
            if manifest.planned_trials != 32400:
                errors.append(f"Manifest planned_trials must be 32400, got {manifest.planned_trials}")

            # Verify DAG structure (no cycles, valid topological sort, strictly intra-group dependencies)
            entry_ids = {e.entry_id for e in manifest.entries}
            if len(entry_ids) != 540:
                errors.append(f"Duplicate entry IDs found: {len(entry_ids)} unique vs 540 total")

            in_degree: Dict[str, int] = {e.entry_id: 0 for e in manifest.entries}
            adjacency: Dict[str, List[str]] = defaultdict(list)
            entry_map: Dict[str, RunManifestEntry] = {e.entry_id: e for e in manifest.entries}

            for e in manifest.entries:
                for dep_id in e.depends_on_entry_ids:
                    if dep_id not in entry_map:
                        errors.append(f"{e.entry_id} depends on unknown entry {dep_id}")
                        continue
                    dep_entry = entry_map[dep_id]
                    # Dependency must belong to same sequence_group_id
                    if dep_entry.sequence_group_id != e.sequence_group_id:
                        errors.append(f"{e.entry_id} depends on entry {dep_id} from different sequence group")
                    # Dependency must have strictly lower sequence_position
                    if dep_entry.sequence_position >= e.sequence_position:
                        errors.append(f"{e.entry_id} (pos {e.sequence_position}) depends on {dep_id} (pos {dep_entry.sequence_position})")

                    adjacency[dep_id].append(e.entry_id)
                    in_degree[e.entry_id] += 1

            # Kahn's algorithm for cycle detection
            queue = deque([eid for eid, deg in in_degree.items() if deg == 0])
            visited_count = 0
            while queue:
                curr = queue.popleft()
                visited_count += 1
                for neighbor in adjacency[curr]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            if visited_count != len(manifest.entries):
                errors.append(f"Cycle detected in execution DAG: visited {visited_count} of {len(manifest.entries)} nodes")

            latency = (time.time() - t0) * 1000.0
            if errors:
                return PreflightCheckResult(
                    gate_number=8,
                    gate_name="Manifest Completeness & DAG Integrity",
                    status="FAILED",
                    details="; ".join(errors[:5]),
                    evidence={"errors_count": len(errors), "first_errors": errors[:5]},
                    latency_ms=latency,
                )

            return PreflightCheckResult(
                gate_number=8,
                gate_name="Manifest Completeness & DAG Integrity",
                status="PASSED",
                details="Catalog completeness and DAG invariants verified: exactly 540 entries across 90 groups (32,400 trials), 0 cycles, strict intra-group predecessor dependencies.",
                evidence={
                    "entry_count": manifest.entry_count,
                    "sequence_group_count": manifest.sequence_group_count,
                    "planned_trials": manifest.planned_trials,
                    "dag_nodes_verified": visited_count,
                    "dag_cycles": 0,
                },
                latency_ms=latency,
            )
        except Exception as e:
            return PreflightCheckResult(
                gate_number=8,
                gate_name="Manifest Completeness & DAG Integrity",
                status="FAILED",
                details=f"Exception during Gate 8 check: {e}",
                evidence={"exception": str(e)},
                latency_ms=(time.time() - t0) * 1000.0,
            )

    # ── Execute All Gates ───────────────────────────────────────────────────────
    def validate_all(self, manifest: Optional[ExecutionManifest] = None) -> PreflightValidationReport:
        """Executes all 8 preflight verification gates sequentially and builds the final report."""
        if manifest is None:
            manifest = generate_execution_manifest(self.cohort_spec, self.sap_spec)

        g1 = self.check_gate1_cohort_integrity()
        g2 = self.check_gate2_sap_integrity()
        g3 = self.check_gate3_materialization_parity()
        g4 = self.check_gate4_transfer_relations(manifest)
        g5 = self.check_gate5_future_leakage_prevention(manifest)
        g6 = self.check_gate6_state_isolation_and_persistence()
        g7 = self.check_gate7_controller_micro_execution()
        g8 = self.check_gate8_manifest_completeness_and_dag(manifest)

        gates = (g1, g2, g3, g4, g5, g6, g7, g8)
        all_passed = all(g.status == "PASSED" for g in gates)

        report_id = f"preflight_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        sap_fp = getattr(self.sap_spec, "plan_fingerprint", getattr(self.sap_spec, "sap_fingerprint", ""))
        report = PreflightValidationReport(
            report_id=report_id,
            report_version="1.0.0",
            cohort_fingerprint=self.cohort_spec.cohort_fingerprint,
            sap_fingerprint=sap_fp,
            manifest_fingerprint=manifest.manifest_fingerprint,
            all_gates_passed=all_passed,
            gates=gates,
        )
        report_fp = report.compute_fingerprint()
        return PreflightValidationReport(
            report_id=report.report_id,
            report_version=report.report_version,
            timestamp=report.timestamp,
            cohort_fingerprint=report.cohort_fingerprint,
            sap_fingerprint=report.sap_fingerprint,
            manifest_fingerprint=report.manifest_fingerprint,
            all_gates_passed=report.all_gates_passed,
            gates=report.gates,
            report_fingerprint=report_fp,
        )

    def generate_attestation(
        self,
        report: PreflightValidationReport,
        cohort_file_sha256: str = "",
        manifest_file_sha256: str = "",
        preflight_report_file_sha256: str = "",
        software_commit: Optional[str] = None,
        dirty_worktree: Optional[bool] = None,
        execution_authorized: bool = False,
        authorized_by: Optional[str] = None,
        authorization_timestamp: Optional[str] = None,
    ) -> PreflightAttestation:
        """Generates a cryptographic preflight attestation token with dual-domain binding.
        
        CRITICAL INVARIANTS:
        1. Binds both canonical object fingerprints and physical file SHA-256 hashes.
        2. Binds exact 40-character Git commit SHA and dirty worktree status.
        3. Anti-circularity: attestation_fingerprint is computed over unsigned payload.
        4. By default, execution_authorized is FALSE.
        """
        if software_commit is None or dirty_worktree is None:
            default_sha, default_dirty = get_software_commit_info()
            commit_sha = software_commit if software_commit is not None else default_sha
            is_dirty = dirty_worktree if dirty_worktree is not None else default_dirty
        else:
            commit_sha = software_commit
            is_dirty = dirty_worktree

        attestation = PreflightAttestation(
            attestation_version="1.1.0",
            cohort_canonical_fingerprint=report.cohort_fingerprint,
            cohort_physical_file_sha256=cohort_file_sha256,
            sap_canonical_fingerprint=report.sap_fingerprint,
            manifest_canonical_fingerprint=report.manifest_fingerprint,
            manifest_physical_file_sha256=manifest_file_sha256,
            preflight_report_canonical_fingerprint=report.report_fingerprint,
            preflight_report_physical_file_sha256=preflight_report_file_sha256,
            software_commit=commit_sha,
            dirty_worktree=is_dirty,
            preflight_pass=report.all_gates_passed,
            execution_authorized=execution_authorized,
            authorized_by=authorized_by,
            authorization_timestamp=authorization_timestamp,
        )
        fp = attestation.compute_fingerprint()
        return PreflightAttestation(
            attestation_version=attestation.attestation_version,
            timestamp=attestation.timestamp,
            cohort_canonical_fingerprint=attestation.cohort_canonical_fingerprint,
            cohort_physical_file_sha256=attestation.cohort_physical_file_sha256,
            sap_canonical_fingerprint=attestation.sap_canonical_fingerprint,
            manifest_canonical_fingerprint=attestation.manifest_canonical_fingerprint,
            manifest_physical_file_sha256=attestation.manifest_physical_file_sha256,
            preflight_report_canonical_fingerprint=attestation.preflight_report_canonical_fingerprint,
            preflight_report_physical_file_sha256=attestation.preflight_report_physical_file_sha256,
            software_commit=attestation.software_commit,
            dirty_worktree=attestation.dirty_worktree,
            preflight_pass=attestation.preflight_pass,
            execution_authorized=attestation.execution_authorized,
            authorized_by=attestation.authorized_by,
            authorization_timestamp=attestation.authorization_timestamp,
            attestation_fingerprint=fp,
        )

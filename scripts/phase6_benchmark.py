"""Phase 6 deterministic microbenchmark for ResearchHistoryService.

Produces a small, deterministic workload and measures:
- apply_events (projection/application) time
- reconstruction time
- consistency-validation time

Writes a JSON `phase6_benchmark_result.json` with machine-readable results.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchforge.domain.state import ResearchState
from researchforge.domain.provenance import Provenance
from researchforge.state.events import Event, EventType
from researchforge.integrator import ResearchHistoryService


def make_prov(pid):
    return Provenance(id=pid, schema_version="1", created_by="bench", created_at="now")


def make_event(eid, etype, payload=None, prov=None):
    return Event.create(id=eid, schema_version="1", event_type=etype, payload=payload or {}, timestamp=None, provenance_id=(prov.id if prov else None))


def build_deterministic_trajectory(cycles=10):
    # fixed initial part
    prov = make_prov("bench-prov")
    evs = []
    evs.append(make_event("b_init_q", EventType.QUESTION_SELECTED, {"question_id": "q_bench", "problem_id": "p_bench"}, prov))
    evs.append(make_event("b_init_h", EventType.HYPOTHESIS_PROPOSED, {"hypothesis_id": "h_bench", "question_id": "q_bench"}, prov))
    evs.append(make_event("b_init_d", EventType.DECISION_MADE, {"decision_id": "d_bench", "hypothesis_id": "h_bench"}, prov))

    # repeated experiment cycles
    for i in range(cycles):
        sid = f"spec_{i}"
        rid = f"run_{i}"
        oid = f"out_{i}"
        evs.append(make_event(f"plan_{i}", EventType.EXPERIMENT_PLANNED, {"spec_id": sid, "decision_id": "d_bench"}, prov))
        evs.append(make_event(f"start_{i}", EventType.EXPERIMENT_STARTED, {"spec_id": sid, "run_id": rid}, prov))
        evs.append(make_event(f"out_{i}", EventType.OUTCOME_RECORDED, {"run_id": rid, "outcome_id": oid}, prov))
        evs.append(make_event(f"val_{i}", EventType.VALIDITY_ASSESSED, {"outcome_id": oid, "verdict": "VALID" if i % 2 == 0 else "INVALID"}, prov))
    return prov, evs


def run_benchmark(cycles=10, result_path="phase6_benchmark_result.json"):
    prov, events = build_deterministic_trajectory(cycles)
    svc = ResearchHistoryService()
    init = ResearchState(id="s_bench", schema_version="1")

    # measure apply_events (includes projection)
    t0 = time.perf_counter()
    final_state = svc.apply_events(init, events, provenance_map={prov.id: prov})
    t1 = time.perf_counter()
    apply_time = t1 - t0

    # measure reconstruction (fresh graph)
    t2 = time.perf_counter()
    final_state2, graph2 = svc.reconstruct(init, events, provenance_map={prov.id: prov})
    t3 = time.perf_counter()
    reconstruct_time = t3 - t2

    # measure consistency validation
    t4 = time.perf_counter()
    rep = svc.validate_consistency()
    t5 = time.perf_counter()
    validate_time = t5 - t4

    result = {
        "events": len(events),
        "nodes": len(svc.graph._nodes),
        "edges": len(svc.graph._edges),
        "apply_time_seconds": apply_time,
        "reconstruct_time_seconds": reconstruct_time,
        "validate_time_seconds": validate_time,
        "report_summary": {
            "num_states": rep.num_states,
            "provenance_records": rep.provenance_records,
            "outcomes": rep.outcomes,
            "failures": rep.failures,
            "consistency": rep.consistency,
        },
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Phase6 microbenchmark result:")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run_benchmark(cycles=10)

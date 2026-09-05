"""Canonical Phase 6 integration example.

Executable demonstration of: ResearchStateTransitionEngine -> ResearchHistoryService -> VRDEG -> query helpers

Run `python3 scripts/phase6_example.py` to execute.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchforge.domain.state import ResearchState
from researchforge.domain.provenance import Provenance
from researchforge.state.events import Event, EventType
from researchforge.integrator import ResearchHistoryService
from researchforge.vrdeg.queries import get_research_trajectory, get_experiment_lineage, get_state_history, get_failure_history, get_related_evidence


def make_prov(pid):
    return Provenance(id=pid, schema_version="1", created_by="example", created_at="now")


def make_event(eid, etype, payload=None, prov=None):
    return Event.create(id=eid, schema_version="1", event_type=etype, payload=payload or {}, timestamp=None, provenance_id=(prov.id if prov else None))


def main():
    svc = ResearchHistoryService()
    init = ResearchState(id="s_ex", schema_version="1")
    pv = make_prov("pv_ex")

    evs = [
        make_event("ex1", EventType.QUESTION_SELECTED, {"question_id": "q_ex", "problem_id": "p_ex"}, pv),
        make_event("ex2", EventType.HYPOTHESIS_PROPOSED, {"hypothesis_id": "h_ex", "question_id": "q_ex"}, pv),
        make_event("ex3", EventType.DECISION_MADE, {"decision_id": "d_ex", "hypothesis_id": "h_ex"}, pv),
        make_event("ex4", EventType.EXPERIMENT_PLANNED, {"spec_id": "spec_ex", "decision_id": "d_ex"}, pv),
        make_event("ex5", EventType.EXPERIMENT_STARTED, {"spec_id": "spec_ex", "run_id": "run_ex"}, pv),
        make_event("ex6", EventType.OUTCOME_RECORDED, {"run_id": "run_ex", "outcome_id": "out_ex"}, pv),
        make_event("ex7", EventType.DIAGNOSIS_RECORDED, {"diagnosis_id": "diag_ex", "outcome_id": "out_ex"}, pv),
    ]

    final = svc.apply_events(init, evs, provenance_map={pv.id: pv})
    report = svc.validate_consistency()

    # queries
    traj = get_research_trajectory(svc.graph, "q_ex")
    lineage = get_experiment_lineage(svc.graph, "spec_ex")
    state_hist = get_state_history(svc.graph, final.id)
    fail_hist = get_failure_history(svc.graph, "nonexistent")

    out = {
        "final_state_fingerprint": final.fingerprint(),
        "graph_fingerprint": svc.graph.fingerprint(),
        "trajectory_nodes": [n.id for n in traj.nodes],
        "lineage_nodes": [n.id for n in lineage.nodes],
        "state_history_nodes": [n.id for n in state_hist.linked_states],
        "failure_history": {"failure": fail_hist.failure.id if fail_hist.failure else None},
        "consistency": report.consistency,
        "integration_report_summary": {
            "num_events": report.num_events,
            "num_nodes": report.num_nodes,
            "num_edges": report.num_edges,
        },
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

from typing import Dict, Any, List
from researchforge.vrdeg.graph import VRDEG, GraphIntegrityError
from researchforge.vrdeg.node import GraphNode, NodeType
from researchforge.vrdeg.edge import GraphEdge, RelationType
from researchforge.state.events import Event, EventType


_id_key_to_node_type = {
    "spec_id": NodeType.EXPERIMENT_SPEC,
    "run_id": NodeType.EXPERIMENT_RUN,
    "outcome_id": NodeType.OUTCOME,
    "failure_id": NodeType.FAILURE,
    "evidence_id": NodeType.EVIDENCE,
    "hypothesis_id": NodeType.HYPOTHESIS,
    "decision_id": NodeType.DECISION,
    "question_id": NodeType.RESEARCH_QUESTION,
    "problem_id": NodeType.RESEARCH_PROBLEM,
    "diagnosis_id": NodeType.DIAGNOSIS,
}


_event_edge_map = {
    EventType.EXPERIMENT_STARTED.value: [("spec_id", "run_id", RelationType.EXECUTED_AS.value)],
    EventType.OUTCOME_RECORDED.value: [("run_id", "outcome_id", RelationType.PRODUCED.value)],
    EventType.EXPERIMENT_PLANNED.value: [("decision_id", "spec_id", RelationType.SELECTED_BY.value)],
    EventType.EXPERIMENT_COMPLETED.value: [],
    EventType.FAILURE_RECORDED.value: [("run_id", "failure_id", RelationType.FAILED_AS.value)],
    EventType.EVIDENCE_ADDED.value: [],
    EventType.HYPOTHESIS_PROPOSED.value: [("question_id", "hypothesis_id", RelationType.MOTIVATED_BY.value)],
    EventType.DECISION_MADE.value: [("hypothesis_id", "decision_id", RelationType.SELECTED_BY.value)],
    EventType.VALIDITY_ASSESSED.value: [],
    EventType.DIAGNOSIS_RECORDED.value: [("outcome_id", "diagnosis_id", RelationType.DIAGNOSED_BY.value)],
    EventType.QUESTION_SELECTED.value: [("problem_id", "question_id", RelationType.ADDRESSES.value)],
    EventType.RESEARCH_INITIALIZED.value: [],
}


class VRDEGProjector:
    """Project validated research events into VRDEG.

    Responsibilities:
    - idempotent projection (use event.id to avoid duplicates)
    - preserve provenance
    - create nodes/edges only when facts are explicit in the event payload
    """

    def __init__(self, graph: VRDEG):
        self.graph = graph

    def _ensure_provenance_node(self, prov_id: str | None, schema_version: str) -> None:
        if not prov_id:
            return
        if not self.graph.get_node(prov_id):
            pn = GraphNode(id=prov_id, schema_version=schema_version, node_type=NodeType.PROVENANCE.value)
            self.graph.add_node(pn)

    def _ensure_node_for_key(self, key: str, val: str, schema_version: str, provenance_id: str | None) -> None:
        node_type = _id_key_to_node_type.get(key)
        if not node_type:
            return
        if not self.graph.get_node(val):
            n = GraphNode(id=val, schema_version=schema_version, node_type=node_type.value, payload_ref=val, provenance_id=provenance_id)
            self.graph.add_node(n)

    def _add_edge_if_missing(self, edge_id: str, schema_version: str, source: str, target: str, relation: str, provenance_id: str | None = None) -> None:
        if self.graph.get_edge(edge_id):
            return
        e = GraphEdge(id=edge_id, schema_version=schema_version, source_id=source, target_id=target, relation=relation, provenance_id=provenance_id)
        self.graph.add_edge(e)

    def project_event(self, event: Event) -> None:
        etype = event.event_type
        payload = event.payload or {}
        # provenance
        self._ensure_provenance_node(event.provenance_id, event.schema_version)

        # create nodes for explicit payload *_id keys
        for k, v in payload.items():
            if k.endswith("_id") and isinstance(v, str):
                self._ensure_node_for_key(k, v, event.schema_version, event.provenance_id)

        # create event-scoped nodes when necessary (e.g., validity assessments)
        if etype == EventType.VALIDITY_ASSESSED.value:
            outcome_id = payload.get("outcome_id")
            verdict = payload.get("verdict")
            if outcome_id and verdict:
                vid = f"validity:{outcome_id}:{verdict}"
                if not self.graph.get_node(vid):
                    n = GraphNode(id=vid, schema_version=event.schema_version, node_type=NodeType.VALIDITY.value, payload_ref=None, provenance_id=event.provenance_id, metadata={"verdict": verdict})
                    self.graph.add_node(n)
                # edge outcome -> validity
                if self.graph.get_node(outcome_id):
                    edge_id = f"edge:{event.id}:validity"
                    self._add_edge_if_missing(edge_id, event.schema_version, outcome_id, vid, RelationType.VALIDATED_BY.value, event.provenance_id)
                return

        # map standard event-driven edges
        mapping = _event_edge_map.get(etype, None)
        if mapping is None:
            raise ValueError(f"unsupported event type for projection: {etype}")

        for src_key, tgt_key, relation in mapping:
            src = payload.get(src_key)
            tgt = payload.get(tgt_key)
            if not src or not tgt:
                continue
            # ensure nodes exist
            self._ensure_node_for_key(src_key, src, event.schema_version, event.provenance_id)
            self._ensure_node_for_key(tgt_key, tgt, event.schema_version, event.provenance_id)
            edge_id = f"edge:{event.id}:{relation}:{src}->{tgt}"
            self._add_edge_if_missing(edge_id, event.schema_version, src, tgt, relation, event.provenance_id)

    def project_events(self, events: List[Event]) -> None:
        for ev in events:
            self.project_event(ev)

    def project_gate_decision(
        self,
        decision_id: str,
        gate_decision: str,
        confidence: float,
        prior_task_id: str,
        target_task_id: str,
        schema_version: str = "1.0.0",
        provenance_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        """Projects an RF-2 transfer gate decision (TRANSFER/SUPPRESS/ABSTAIN) into VRDEG provenance edge store."""
        self._ensure_provenance_node(provenance_id, schema_version)
        if not self.graph.get_node(decision_id):
            dn = GraphNode(
                id=decision_id,
                schema_version=schema_version,
                node_type=NodeType.DECISION.value,
                payload_ref=decision_id,
                provenance_id=provenance_id,
            )
            self.graph.add_node(dn)

        if not self.graph.get_node(target_task_id):
            tn = GraphNode(
                id=target_task_id,
                schema_version=schema_version,
                node_type=NodeType.RESEARCH_PROBLEM.value,
                payload_ref=target_task_id,
                provenance_id=provenance_id,
            )
            self.graph.add_node(tn)

        edge_id = f"edge:gate:{decision_id}->{target_task_id}"
        if not self.graph.get_edge(edge_id):
            edge = GraphEdge(
                id=edge_id,
                schema_version=schema_version,
                source_id=decision_id,
                target_id=target_task_id,
                relation=RelationType.GATED_BY.value,
                provenance_id=provenance_id,
                confidence=confidence,
                metadata={
                    "gate_decision": gate_decision,
                    "prior_task_id": prior_task_id,
                    **(metadata or {}),
                },
            )
            self.graph.add_edge(edge)

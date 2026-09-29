import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Dict, Any
from researchforge.benchmarks.continuity.tasks import (
    digits_all_task, 
    digits_0_4_task, 
    digits_5_9_task, 
    digits_shifted_task, 
    synthetic_ecg_lead1_task,
    tabular_xor_parity_task, 
    capacity_trap_task
)
from researchforge.pipeline.controller import ResearchController

def transfer_state(ctrl_source: ResearchController, ctrl_target: ResearchController):
    # Transfer memory state
    ctrl_target.ecrm = ctrl_source.ecrm
    ctrl_target._posterior_memory = ctrl_source._posterior_memory
    ctrl_target._contextual_bandit = ctrl_source._contextual_bandit
    
    # Transfer Phase 12 state
    ctrl_target._utility_predictor = ctrl_source._utility_predictor
    ctrl_target._evidence_service = ctrl_source._evidence_service
    ctrl_target._evidence_service._utility_predictor = ctrl_target._utility_predictor.get_predictor_callable()

def run_b3():
    print("--- BENCHMARK 3: THE TRANSFER MATRIX ---")
    
    seed = 42
    n_gens_source = 20
    n_gens_target = 15
    cond = "contextual_policy"
    
    # 1. Source Task (T1)
    print("\n[Phase 1] Training Source: digits_all")
    task_source = digits_all_task(seed=seed).task
    ctrl_source = ResearchController(task_source, condition=cond, seed=seed)
    res_source = ctrl_source.run(n_generations=n_gens_source)
    print(f"Source Final Metric: {res_source.best_metric:.4f}")
    
    # 2. Target Tasks
    targets = [
        ("T2", "digits_0_4 (Same-family, Positive)", digits_0_4_task(seed=seed).task),
        ("T3", "digits_5_9 (Same-family, Positive)", digits_5_9_task(seed=seed).task),
        ("T4", "digits_shifted (Same-family, Shifted)", digits_shifted_task(seed=seed).task),
        ("T5", "synthetic_ecg (Different family)", synthetic_ecg_lead1_task(seed=seed).task),
        ("T6", "tabular_xor (Different family)", tabular_xor_parity_task(seed=seed).task),
        ("T7", "capacity_trap (Harmful)", capacity_trap_task(seed=seed).task),
    ]
    
    results = {}
    
    print("\n[Phase 2] Evaluating Targets")
    for tid, name, task_target in targets:
        print(f"\n================ {tid}: {name} ================")
        
        # Cold Start Baseline
        ctrl_cold = ResearchController(task_target, condition=cond, seed=seed)
        res_cold = ctrl_cold.run(n_generations=n_gens_target)
        
        # Transfer Run
        ctrl_transfer = ResearchController(task_target, condition=cond, seed=seed)
        transfer_state(ctrl_source, ctrl_transfer)
        res_trans = ctrl_transfer.run(n_generations=n_gens_target)
        
        delta = res_trans.best_metric - res_cold.best_metric
        
        results[tid] = {
            "name": name,
            "cold_start": res_cold.best_metric,
            "transfer": res_trans.best_metric,
            "delta": delta,
        }
        
        print(f"  Cold Start: {res_cold.best_metric:.4f}")
        print(f"  Transfer:   {res_trans.best_metric:.4f}  (Delta: {delta:+.4f})")
        
        # Extract gate decisions from the final generation of transfer
        if ctrl_transfer._last_decision_metadata:
            contrib = ctrl_transfer._last_decision_metadata.get("memory_decision_contribution", {})
            gate_decisions = contrib.get("gate_decisions", {})
            if gate_decisions:
                print("  Gate Decisions Sample:")
                # just print 3 decisions to keep output clean
                for i, (action, dec) in enumerate(gate_decisions.items()):
                    if i > 2: break
                    gate = dec.get('gate_action', 'UNKNOWN')
                    mod = dec.get('modifier', 0.0)
                    print(f"    {action[:15]:15s} -> {gate:9s} [Mod:{mod:+.3f}]")

    print("\n--- TRANSFER MATRIX SUMMARY ---")
    for tid, data in results.items():
        print(f"{tid} ({data['name'][:20]:20s}): Delta {data['delta']:+.4f}")

if __name__ == "__main__":
    run_b3()

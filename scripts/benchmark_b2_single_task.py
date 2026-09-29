import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List
from researchforge.benchmarks.continuity.tasks import digits_all_task, synthetic_ecg_lead1_task, tabular_xor_parity_task, capacity_trap_task
from researchforge.pipeline.controller import ResearchController

def run_b2():
    print("--- BENCHMARK 2: SINGLE-TASK POLICY BEHAVIOR ---")
    
    # We want tasks: digits, synthetic_ecg, XOR, capacity_trap
    tasks = [
        digits_all_task(seed=42).task,
        synthetic_ecg_lead1_task(seed=42).task,
        tabular_xor_parity_task(seed=42).task,
        capacity_trap_task(seed=42).task,
    ]
    
    conditions = ["no_memory", "full", "contextual_policy"]
    n_gens = 10
    seed = 42
    
    print(f"Tasks to run: {[t.name for t in tasks]}")
    
    for task in tasks:
        print(f"\n================ TASK: {task.name} ================")
        
        for cond in conditions:
            print(f"\nCondition: {cond}")
            ctrl = ResearchController(task, condition=cond, seed=seed)
            res = ctrl.run(n_generations=n_gens)
            
            print(f"  Best Metric: {res.best_metric:.4f}")
            
            if cond == "contextual_policy" and ctrl._last_decision_metadata:
                print("  Decision Trace (Final Gen):")
                contrib = ctrl._last_decision_metadata.get("memory_decision_contribution", {})
                gate_decisions = contrib.get("gate_decisions", {})
                print(f"  Got {len(gate_decisions)} gate decisions.")
                for action, dec in gate_decisions.items():
                    action_str = f"{action[:20]:20s}"
                    gate = dec.get('gate_action', 'UNKNOWN')
                    p_b = dec.get('p_benefit', 0.0)
                    p_n = dec.get('p_neutral', 0.0)
                    p_h = dec.get('p_harm', 0.0)
                    mod = dec.get('modifier', 0.0)
                    
                    if gate != 'ATTENUATE' or mod != 0.0:
                        print(f"    {action_str} -> {gate:9s} (P_B:{p_b:.2f}, P_N:{p_n:.2f}, P_H:{p_h:.2f}) [Mod:{mod:+.3f}]")
                
                nodes_list = list(ctrl.rdg.nodes.values())
                if len(nodes_list) >= 2:
                    print("  Final chosen strategy:", nodes_list[-2].attributes.get('strategy', 'unknown'))

if __name__ == "__main__":
    run_b2()

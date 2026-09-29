"""researchforge/discovery/bandit_expansion.py — Dynamic Action Injection.

Demonstrates how the MetaController injects a newly compiled DSL operator
into a live ResearchController run, dynamically expanding the bandit's arms.
"""
import sys
import os
import random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from researchforge.benchmarks.continuity.tasks import digits_all_task
from researchforge.pipeline.controller import ResearchController
from researchforge.discovery.operator_genome import create_increase_capacity_operator
from researchforge.discovery.dsl_compiler import DSLCompiler
from researchforge.genome.operators import register_dynamic_strategy

def run_bandit_expansion_demo():
    print("--- RF-3 Algorithm Discovery: Dynamic Bandit Expansion ---")
    
    # 1. Instantiate the live controller
    seed = 42
    task = digits_all_task(seed=seed).task
    ctrl = ResearchController(task, condition="contextual_policy", seed=seed)
    
    print(f"[Init] Bandit action space size: {len(ctrl._contextual_bandit.actions)}")
    
    # 2. Run a few generations with the default operators
    print("[Run] Executing 5 generations with default operators...")
    ctrl.run(n_generations=5)
    
    # 3. Simulate MetaController discovering and compiling a new algorithm
    print("\n[MetaController] Evolving a new structural algorithm...")
    compiler = DSLCompiler()
    
    # We mutate the 'increase_capacity' operator to make something slightly different
    base_op = create_increase_capacity_operator()
    # Mutate to create "Evolved_..." operator
    rng = random.Random(99)
    evolved_op = base_op.mutate(rng)
    
    print(f"  -> Discovered: {evolved_op.name}")
    print(f"  -> Primitives: {[p.name for p in evolved_op.primitives]}")
    
    # Compile to Python function
    compiled_func = compiler.compile(evolved_op)
    
    # 4. Inject into the ecosystem
    print("\n[Injection] Registering the new algorithm into the live environment...")
    register_dynamic_strategy(evolved_op.name, compiled_func)
    
    # 5. Expand the bandit's arms dynamically
    ctrl._contextual_bandit.add_action(evolved_op.name)
    
    print(f"[Update] Bandit action space size is now: {len(ctrl._contextual_bandit.actions)}")
    print(f"         New action added: {evolved_op.name}")
    
    # 6. Resume the run; the bandit can now pull the new arm!
    print("\n[Run] Resuming search for 5 more generations. The bandit can now select the new algorithm!")
    # Just to force some exploration of the new arm, we can inject artificial uncertainty 
    # but the UCB formula naturally explores new arms because _n_updates is 0 and variance is high!
    res = ctrl.run(n_generations=5)
    
    print(f"\n[Complete] Final Metric: {res.best_metric:.4f}")
    
    # Check if the bandit actually tried it
    trials = ctrl._contextual_bandit._n_updates.get(evolved_op.name, 0)
    print(f"The contextual bandit evaluated '{evolved_op.name}' {trials} times during the final 5 generations.")
    
if __name__ == "__main__":
    run_bandit_expansion_demo()

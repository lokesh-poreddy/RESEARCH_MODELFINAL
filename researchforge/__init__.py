"""ResearchForge-ECRM: a reference implementation of the architecture described in

    ResearchForge-ECRM: An Evidence- and Outcome-Conditioned Research
    Development Graph for Autonomous Model Evolution and Algorithm Discovery

This package implements, and actually runs, the core loop from that design:

    select branch (Policy Learner)
        -> synthesize a Model Genome (Discovery Pipeline / evolution operators)
        -> run the experiment (Evaluator)
        -> diagnose the outcome (Failure Taxonomy)
        -> write to memory (ECRM)
        -> update the policy
        -> repeat

on top of a persistent, typed Research Development Graph (RDG).

See README.md for what is a faithful implementation of the design document
and what is a deliberately lightweight, offline stand-in for infrastructure
(Neo4j, Postgres+pgvector, an LLM encoder, a GPU cluster) this environment
cannot provide.
"""

__version__ = "1.0.0"  # RF-1 + RF-2 (contextual transfer policy) + RF-3 (algorithm discovery)

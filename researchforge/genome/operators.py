"""Evolution Operators acting on Model Genomes: parameter mutation, structure
mutation, crossover, and family-level "algorithm discovery" resets -- matching
ResearchForge-ECRM Sec. 4.5 (pseudocode) and Sec. 5 of the technical report.

Each function takes a parent genome and returns a *new* genome (parents are
never mutated in place, matching the RDG's DERIVES_FROM/parent_ids lineage
tracking). Numeric safety is handled centrally in
`model_genome.ModelGenome.build_estimator()`, so these operators are free to
explore aggressively -- a bad draw produces a weak model to be diagnosed and
selected against, not a crash.
"""
from __future__ import annotations
import random
from typing import List, Optional

from .model_genome import ModelGenome, DEFAULT_GENOMES

# The "branches" the Policy Learner chooses between (Sec. 4.1 select_branch).
# Each corresponds to a qualitative research strategy, not just a knob turn --
# consistent with the design doc's Strategy nodes / SeaEvo-style strategy
# descriptions (Sec. 2).
STRATEGIES = [
    "increase_capacity",
    "add_regularization",
    "tune_learning_dynamics",
    "change_family",
    "feature_preprocessing",
    "crossover_top2",
    "random_perturbation",
]

# Which hyperparameters `param_mutation`/`random_perturbation` are allowed to
# perturb for each model family.
NUMERIC_PARAM_MAP = {
    "MLPClassifier": ["alpha", "learning_rate_init"],
    "RandomForestClassifier": ["min_samples_leaf", "min_samples_split"],
    "SVC": ["C"],
    "LogisticRegression": ["C"],
}


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def param_mutation(genome: ModelGenome, rng: random.Random, delta: float = 0.4) -> ModelGenome:
    """Point mutation: perturb one numeric hyperparameter by +/- delta
    (design doc Sec. 4.5 `mutate_model` / Sec. 5 `param_mutation`)."""
    g = genome.clone()
    keys = NUMERIC_PARAM_MAP.get(g.model_type, [])
    if not keys:
        return g
    key = rng.choice(keys)
    val = g.hyperparameters.get(key)
    if val is None:
        return g
    g.hyperparameters[key] = max(1e-6, val * (1 + rng.uniform(-delta, delta)))
    g.parent_ids = [genome.model_id]
    return g


def increase_capacity(genome: ModelGenome, rng: random.Random) -> ModelGenome:
    """Structure mutation towards more capacity: bigger hidden layers, more
    trees / deeper trees, larger C (less-regularized margin)."""
    g = genome.clone()
    if g.model_type == "MLPClassifier":
        sizes = list(g.architecture.get("hidden_layer_sizes", [64]))
        sizes[-1] = int(_clip(sizes[-1] * rng.uniform(1.3, 2.0), 8, 512))
        if rng.random() < 0.35:
            sizes.append(int(_clip(sizes[-1] / 2, 8, 256)))
        g.architecture["hidden_layer_sizes"] = sizes
    elif g.model_type == "RandomForestClassifier":
        g.architecture["n_estimators"] = int(_clip(
            g.architecture.get("n_estimators", 100) * rng.uniform(1.2, 1.6), 10, 600))
        depth = g.architecture.get("max_depth")
        g.architecture["max_depth"] = None if depth is None else int(_clip(depth * 1.3, 2, 64))
    elif g.model_type in ("SVC", "LogisticRegression"):
        g.hyperparameters["C"] = _clip(g.hyperparameters.get("C", 1.0) * rng.uniform(1.5, 3.0), 1e-3, 1e4)
    g.parent_ids = [genome.model_id]
    return g


def add_regularization(genome: ModelGenome, rng: random.Random) -> ModelGenome:
    """Structure/parameter mutation towards less capacity: smaller effective
    margin / bigger leaves / stronger L2 penalty."""
    g = genome.clone()
    if g.model_type == "MLPClassifier":
        g.hyperparameters["alpha"] = _clip(
            g.hyperparameters.get("alpha", 1e-4) * rng.uniform(2, 6), 1e-6, 10.0)
    elif g.model_type == "RandomForestClassifier":
        g.hyperparameters["min_samples_leaf"] = int(_clip(
            g.hyperparameters.get("min_samples_leaf", 1) + rng.randint(1, 3), 1, 30))
    elif g.model_type in ("SVC", "LogisticRegression"):
        g.hyperparameters["C"] = _clip(g.hyperparameters.get("C", 1.0) / rng.uniform(1.5, 3.0), 1e-3, 1e4)
    g.parent_ids = [genome.model_id]
    return g


def tune_learning_dynamics(genome: ModelGenome, rng: random.Random) -> ModelGenome:
    """Mutation targeting *how* the model is fit rather than its capacity:
    learning rate, RF split granularity, SVC kernel coefficient, iteration budget."""
    g = genome.clone()
    if g.model_type == "MLPClassifier":
        g.hyperparameters["learning_rate_init"] = _clip(
            g.hyperparameters.get("learning_rate_init", 1e-3) * rng.uniform(0.3, 3.0), 1e-5, 1.0)
    elif g.model_type == "SVC":
        g.hyperparameters["gamma"] = rng.choice(["scale", "auto"])
    elif g.model_type == "RandomForestClassifier":
        g.hyperparameters["min_samples_split"] = int(_clip(
            g.hyperparameters.get("min_samples_split", 2) + rng.choice([-1, 1]), 2, 20))
    else:
        g.hyperparameters["max_iter"] = int(_clip(
            g.hyperparameters.get("max_iter", 2000) * rng.uniform(0.5, 2.0), 50, 5000))
    g.parent_ids = [genome.model_id]
    return g


def change_family(genome: ModelGenome, rng: random.Random) -> ModelGenome:
    """Algorithm-Discovery-style operator: abandon the current model family
    for a different one, carrying over only the data pipeline (design doc
    Sec. 1 'Algorithm-Discovery Mechanism' -- triggered when a branch plateaus)."""
    others = [m for m in DEFAULT_GENOMES if m != genome.model_type]
    new_type = rng.choice(others)
    g = ModelGenome.default(new_type, seed=genome.seed)
    g.data_pipeline = dict(genome.data_pipeline)
    g.generation = genome.generation
    g.parent_ids = [genome.model_id]
    return g


def feature_preprocessing(genome: ModelGenome, rng: random.Random) -> ModelGenome:
    """Mutation over the data_pipeline field: toggle scaling, try PCA."""
    g = genome.clone()
    g.data_pipeline["scale"] = not g.data_pipeline.get("scale", True)
    if rng.random() < 0.5:
        g.data_pipeline["pca_components"] = rng.choice([None, 0.9, 0.95, 16, 32])
    g.parent_ids = [genome.model_id]
    return g


def crossover(genome_a: ModelGenome, genome_b: ModelGenome, rng: random.Random) -> ModelGenome:
    """Combine two parent genomes (design doc Sec. 4.5 `crossover`). Same
    family: mix hyperparameters/architecture field-by-field. Different
    families: keep one parent's model but may borrow the other's data pipeline."""
    if genome_a.model_type != genome_b.model_type:
        base, other = (genome_a, genome_b) if rng.random() < 0.5 else (genome_b, genome_a)
        g = base.clone()
        if rng.random() < 0.5:
            g.data_pipeline = dict(other.data_pipeline)
        g.parent_ids = [genome_a.model_id, genome_b.model_id]
        return g
    g = genome_a.clone()
    for key in set(genome_a.hyperparameters) | set(genome_b.hyperparameters):
        if key in genome_b.hyperparameters and rng.random() < 0.5:
            g.hyperparameters[key] = genome_b.hyperparameters[key]
    for key in set(genome_a.architecture) | set(genome_b.architecture):
        if key in genome_b.architecture and rng.random() < 0.5:
            g.architecture[key] = genome_b.architecture[key]
    g.parent_ids = [genome_a.model_id, genome_b.model_id]
    return g


def random_perturbation(genome: ModelGenome, rng: random.Random) -> ModelGenome:
    """The 'no real strategy' operator: one or two blind parameter mutations.
    Used as the sole operator in the random-search RDE-Bench baseline, and as
    one of the 7 branches the learned policy can still pick under the full system."""
    g = genome
    for _ in range(rng.randint(1, 2)):
        g = param_mutation(g, rng, delta=0.6)
    return g


# Global registry for dynamically compiled DSL operators
DYNAMIC_STRATEGIES = {}

def register_dynamic_strategy(name: str, func: Any) -> None:
    """Registers a new algorithm discovered by MetaController."""
    DYNAMIC_STRATEGIES[name] = func
    if name not in STRATEGIES:
        STRATEGIES.append(name)

def apply_strategy(strategy: str, genome: Any, rng: random.Random,
                    population: Optional[List[Any]] = None) -> Any:
    """Strategy name -> concrete genome edit. Works with ModelGenome and TargetModelGenome."""
    from .target_model_genome import TargetModelGenome
    is_tmg = isinstance(genome, TargetModelGenome)
    g = genome.to_model_genome() if is_tmg else genome
    pop = [p.to_model_genome() if isinstance(p, TargetModelGenome) else p for p in population] if population else None

    if strategy in DYNAMIC_STRATEGIES:
        res = DYNAMIC_STRATEGIES[strategy](g)
    elif strategy == "increase_capacity":
        res = increase_capacity(g, rng)
    elif strategy == "add_regularization":
        res = add_regularization(g, rng)
    elif strategy == "tune_learning_dynamics":
        res = tune_learning_dynamics(g, rng)
    elif strategy == "change_family":
        res = change_family(g, rng)
    elif strategy == "feature_preprocessing":
        res = feature_preprocessing(g, rng)
    elif strategy == "crossover_top2":
        if pop and len(pop) >= 2:
            a, b = rng.sample(pop, 2)
            res = crossover(a, b, rng)
        else:
            res = random_perturbation(g, rng)
    elif strategy == "random_perturbation":
        res = random_perturbation(g, rng)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'")

    if is_tmg:
        tmg = TargetModelGenome.from_model_genome(res)
        tmg.operator = strategy
        return tmg
    return res

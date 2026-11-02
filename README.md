# ResearchForge-ECRM (v1.0.0)

An Evidence- and Outcome-Conditioned Research Development Graph for Autonomous Model Evolution and Algorithm Discovery.

ResearchForge-ECRM unifies three generations of autonomous AI research systems into a production-hardened, self-contained architecture:
1. **RF-1:** The foundational evolutionary controller, typed Research Development Graph (RDG), and Evidence- and Outcome-Conditioned Research Memory (ECRM).
2. **RF-2:** Contextual Transfer Policy driven by a LinUCB contextual bandit, Bayesian linear regression utility predictor, and utility-aware transfer gating.
3. **RF-3:** Autonomous Algorithm Discovery powered by an AST/DSL compiler, evolutionary meta-controller, and dynamic bandit arm expansion.

---

## Architecture Overview

```
Task / Dataset
   │
   ▼
ResearchController (pipeline/controller.py)
   ├── ContextualBandit (policy/contextual_bandit.py - LinUCB)
   │       ├── Built-in Heuristic Strategies (7 canonical arms)
   │       └── DSL-Compiled Discovered Algorithms (DYNAMIC_STRATEGIES registry)
   │
   ├── ModelGenome / TargetModelGenome (genome/target_model_genome.py)
   │       └── Evaluator (evaluators/sklearn_evaluator.py)
   │
   ├── FailureTaxonomy (diagnosis/failure_taxonomy.py)
   │       └── Diagnoses failure modes (overfitting, underfitting, divergence, capacity)
   │
   ├── ECRM + PosteriorMemory (memory/ecrm.py, memory/posterior.py)
   │       └── Hierarchical memory consolidation & state persistence
   │
   ├── TransferUtilityPredictor (transfer/utility_learner.py)
   │       └── UtilityAwareGate (policy/utility_gate.py)
   │               └── Probability-driven gating: TRANSFER, SUPPRESS, ABSTAIN
   │
   └── CounterfactualArbitrator (transfer/counterfactual.py)
           └── CounterfactualSamplingPolicy (transfer/sampling.py)
                   └── Paired trial observation and active epsilon-sampling

MetaController (discovery/meta_controller.py - RF-3)
   └── OperatorGenome Evolution Population
           └── DSLCompiler (discovery/dsl_compiler.py)
                   └── register_dynamic_strategy() -> Live Bandit Arm Injection
```

---

## Quickstart

### Installation

```bash
git clone https://github.com/lokesh-poreddy/RESEARCH_MODELFINAL.git
cd RESEARCH_MODELFINAL
pip install -r requirements.txt
```

### Running Unit Tests

```bash
# Run isolated RF-3 Discovery & Counterfactual test suites (28 tests)
python3 -m pytest -p no:cov -o testpaths="" tests/test_rf3_discovery.py tests/test_phase13_counterfactual.py -v

# Run full test suite
python3 -m pytest -p no:cov -q --tb=short --ignore=tests/test_hf_live.py
```

### Running the Live RF-3 Dynamic Bandit Expansion Pilot

```bash
python3 scripts/rf3_bandit_expansion_demo.py
```

### Running the Benchmarks

```bash
python3 scripts/benchmark_b1_regression.py     # B1: Regression benchmark
python3 scripts/benchmark_b3_transfer_matrix.py # B3: Multi-task transfer matrix
python3 scripts/benchmark_b10_ablation.py      # B10: Ablation study with Cohen's d
python3 scripts/run_chemreact_experiment.py    # ChemReact: Reaction-kinetics defect & titration benchmark
```

### Launching the REST API

```bash
uvicorn researchforge.api.server:app --reload --host 0.0.0.0 --port 8000
```

---

## Empirical Benchmark and Testing Results

### A. RF-3 Discovery and Counterfactual Tests
- **Test Modules:** `tests/test_rf3_discovery.py` and `tests/test_phase13_counterfactual.py`
- **Result:** 28 / 28 Passed (100% Pass Rate), 0 Failures, 0 Warnings.
- **Scope Verified:**
  - Primitive AST mutation and parameter noise generation.
  - Polymorphic DSL compiler handlers executing against both dictionary genomes and typed `ModelGenome` objects.
  - Strict architecture safety clamping (`_MIN_UNITS = 8` to `_MAX_UNITS = 2048`, `_MIN_LAYERS = 1` to `_MAX_LAYERS = 10`).
  - Unknown primitive tolerance with fail-soft execution.
  - Live dynamic strategy registration and closure immutability.
  - Tri-state counterfactual arbitration (`COUNTERFACTUAL_OBSERVED`, `COUNTERFACTUAL_ESTIMATED`, `COUNTERFACTUAL_UNAVAILABLE`).

### B. Benchmark B1: RF-2 Regression Benchmark (3 seeds x 25 generations)
Evaluates whether adding contextual bandit gating degrades baseline RF-1 evolutionary performance:

| Condition | Avg Best Metric | Std Dev | Running-Best AUC | TTT @ 0.90 Target | Status |
|---|---|---|---|---|---|
| `no_memory` | 0.8779 | 0.0189 | 0.8741 | 19.3 gens | Baseline |
| `random` | 0.8752 | 0.0198 | 0.8696 | 21.7 gens | Baseline |
| `full` (RF-1 Baseline) | 0.8825 | 0.0132 | 0.8773 | 19.3 gens | Reference |
| `contextual_policy` (RF-2) | 0.8779 | 0.0267 | 0.8728 | 20.7 gens | [PASS] (Threshold >= 0.8648) |

*Evaluation verdict:* RF-2 achieves 99.48% of the RF-1 reference metric, comfortably exceeding the >= 98% non-regression acceptance threshold.

### C. Benchmark B3: Multi-Task Transfer Matrix (Tasks T2 - T7)
Measures transfer delta: Delta = Metric(transfer) - Metric(cold) across distinct task regimes:

| Target Task | Regime / Description | Cold Metric | Transfer Metric | Delta (Delta) | Policy Decision |
|---|---|---|---|---|---|
| T2 (`digits_0_4`) | Low-digits classification | 0.9628 | 0.9163 | -0.0465 | SUPPRESS |
| T3 (`digits_5_9`) | High-digits classification | 0.9019 | 0.9019 | +0.0000 | ABSTAIN (Neutral) |
| T4 (`digits_shifted`) | Perturbed pixel morphology | 0.5933 | 0.5507 | -0.0426 | SUPPRESS |
| T5 (`synthetic_ecg`) | Continuous signal waveform | 0.6903 | 0.7323 | +0.0420 | TRANSFER (Positive Lift) |
| T6 (`tabular_xor`) | Orthogonal parity problem | 0.5259 | 0.8778 | +0.3519 | TRANSFER (Massive OOD Lift) |
| T7 (`capacity_trap`) | Constrained capacity trap | 0.7963 | 0.6333 | -0.1630 | SUPPRESS (Harm Avoided) |

### D. Benchmark B10: Ablation Study (Cohen's d vs Full RF-2)
Effect-size statistical comparison of architectural components:

| Condition | Configuration | Avg Best | AUC | Cohen's d vs Full RF-2 | Interpretation |
|---|---|---|---|---|---|
| Condition A | Full RF-2 (LinUCB + Posterior Gate + Sampling) | 0.8744 | 0.8683 | 0.000 | Baseline |
| Condition B | RF-2 without Counterfactual Sampling | 0.8652 | 0.8642 | +0.399 | Moderate Degradation without active sampling |
| Condition C | RF-1 Trajectory Memory | 0.8763 | 0.8721 | -0.074 | Statistical Equivalence |
| Condition D | Uniform Random Search | 0.8737 | 0.8666 | +0.029 | Neutral / Unfocused exploration |

### E. Live RF-3 Dynamic Bandit Expansion Pilot Run
Executed via `scripts/rf3_bandit_expansion_demo.py`:
- **Initial Setup:** The LinUCB contextual bandit started with 7 standard heuristic operator arms (`increase_capacity`, `add_regularization`, `tune_learning_dynamics`, `change_family`, `feature_preprocessing`, `crossover_top2`, `random_perturbation`).
- **Autonomous Evolution:** The MetaController performed mutation search over AST primitives and discovered `Evolved_increase_capacity_dsl_643` composed of primitives `['ExpandWidth', 'ExpandWidth']`.
- **Dynamic Compilation & Registration:** Compiled into an executable callable via `DSLCompiler` and registered into the global `DYNAMIC_STRATEGIES` catalog.
- **Bandit Arm Expansion:** Injected `Evolved_increase_capacity_dsl_643` into the active contextual bandit, expanding the action space from K = 7 to K = 8 actions without restarting the experiment.
- **Autonomous Search Convergence:** Search resumed for 5 generations with the expanded operator set, achieving a final validation metric of 0.8629.

### F. ChemReact: Reaction-Kinetics Defect Detection & Titration Benchmark
Full paradigm specification and report in `docs/CHEMREACT_RESEARCH_REPORT.md` and results in `chemreact_evaluation_results.json`:
- **Mathematical Paradigm:** Models defect formation via Arrhenius reaction kinetics:
  $$k_f = A \exp\left(-\frac{E_a}{R T_{\text{eff}}}\right), \quad r_f = k_f [C]^\alpha [D]^\beta [K]^\gamma$$
  and automated test-case generation as chemical affinity neutralization titration reaching equivalence:
  $$\kappa_{ij} = \exp\left(-\frac{\Delta G^\ddagger(t_i, d_j)}{k_B T_{\text{eff}}}\right)$$
- **Defect Detection Evaluation (N=600, 12 kinetic features, target=0.88):**
  - Validation Accuracy: **96.81%** (contextual_policy: **96.67%** +/- 0.0000)
  - Held-out Test Accuracy: **94.31% - 96.25%**
  - Held-out Test ROC-AUC: **0.9903 - 0.9930**
  - Held-out Test F1 Score: **0.9420 - 0.9615**
  - Failure Repetition Rate (FRR): **0.0000** (vs 0.0208 for random search)
- **Multi-Task Test-Case Titration Equivalence (N=500, 14 features, target=0.85):**
  - Test Accuracy: **92.50%**
  - Test ROC-AUC: **0.9817**
  - Test F1 Score: **0.9351**
- **Autonomous RF-3 Operator Discovery on ChemReact:**
  - Evolved and compiled `CatalyticArrheniusExpansion` (`ExpandWidth`, `InsertResidual`, `AddLayer`).
  - Dynamically injected into the live contextual bandit during execution with zero runtime interruptions.

---

## Trained and System Configurations

### A. LinUCB Contextual Bandit Configuration
- **Context Dimension (d = 9):**
  1. `task_metric_target`: Target performance bound normalized to [0, 1].
  2. `sample_count_log`: log(1 + N_train_samples).
  3. `feature_dimension_log`: log(1 + d_features).
  4. `generation_progress`: Linear normalized stage t / T.
  5. `current_best_metric`: Best metric observed so far in the run.
  6. `stagnation_counter`: Consecutive generations without metric improvement.
  7. `capacity_bucket`: Categorical model parameter density tier (0: small, 1: medium, 2: large).
  8. `last_failure_category`: Encoded failure category from the diagnostic taxonomy.
  9. `historical_strategy_success_rate`: Running success frequency for the candidate strategy.
- **Exploration Coefficient:** alpha = 0.25 (UCB confidence radius).
- **Ridge Regularization:** lambda = 1.0 (initialization matrix A_a = I_d).
- **Decision Formula:**
  score(a) = (x^T * theta_hat_a + alpha * sqrt(x^T * A_a^-1 * x)) * multiplier_gate

### B. Bayesian Transfer Utility Predictor and Gate
- **Conjugate Prior:** Normal-Inverse-Gamma distribution.
- **Hyperparameters:** Update learning rate beta = 10.0, prior precision alpha_0 = 1.0.
- **Feature Normalization:** log(1 + N_observations) scaling to prevent large trial sample counts from dominating covariance matrix inversion.
- **Tri-State Gating Bounds:**
  - P_benefit = P(Delta > +0.01)
  - P_harm = P(Delta < -0.01)
  - P_neutral = 1 - (P_benefit + P_harm)
- **Gate Decision Rules & Arm Modifiers:**
  - `TRANSFER`: If P_benefit >= 0.65 -> LinUCB score multiplier = 1.2
  - `SUPPRESS`: If P_harm >= 0.50 -> LinUCB score multiplier = 0.5
  - `ABSTAIN`: Otherwise -> LinUCB score multiplier = 1.0

### C. Active Counterfactual Sampling Schedule
- **Initial Epsilon:** epsilon_0 = 0.20 (20% forced paired exploration at generation 0 to overcome cold-start bias).
- **Epsilon Decay Floor:** Exponential decay toward epsilon_min = 0.01 at generation T.
- **Arbitration States:**
  - `COUNTERFACTUAL_OBSERVED`: Empirical paired run delta observed directly (variance = 0.0).
  - `COUNTERFACTUAL_ESTIMATED`: Validated surrogate Gaussian prediction with 95% confidence intervals.
  - `COUNTERFACTUAL_UNAVAILABLE`: Fallback to evidence gating when surrogate variance exceeds bounds.

### D. Model Genome Architecture and Search Bounds
- **Supported Families:** `MLPClassifier`, `RandomForestClassifier`, `SVC`, `LogisticRegression`.
- **Architectural Bounds:**
  - Layer Width: Minimum = 8 units, Maximum = 2048 units.
  - Layer Depth: Minimum = 1 hidden layer, Maximum = 10 hidden layers.
  - Learning Rate: Minimum = 0.00001, Maximum = 0.5, Initial default = 0.01.
  - Regularization Parameter (alpha / C): Range [10^-6, 10.0].

### E. ECRM Memory and Sandbox Execution Limits
- **Retention Decay Policy:** P_retain = exp(-lambda * age), where lambda = 0.05.
- **Analytic Memory Half-Life:** ln(2) / lambda approx 13.86 generations.
- **Consolidation Gate:** `retention_threshold = 0.5`. Records surviving consolidation cycles graduate from Short-Term into Long-Term vector storage.
- **`SafeRunner` Resource Budgets:**
  - Maximum cumulative experiment runs: 1,000 runs
  - Maximum cumulative wall-clock budget: 1,800.0 seconds
  - Per-experiment timeout: 15.0 seconds (enforced via process kill signal)

---

## REST API Specification

The API decouples transport contracts from internal domain logic using structured DTOs and deterministic error envelopes:

| Endpoint | Method | Description |
|---|---|---|
| `/api/discovery/meta-generation` | POST | Executes one evolutionary cycle of the MetaController and returns promoted algorithms |
| `/api/discovery/operators` | GET | Lists all dynamically compiled and registered mutation strategies in the live registry |
| `/experiments` | POST | Submits a new experiment specification |
| `/experiments/{id}/run` | POST | Dispatches experiment execution within the isolated `SafeRunner` sandbox |
| `/models/mutate` | POST | Applies a mutation strategy to a target model genome |
| `/memory/query` | GET | Queries ECRM memory via vector similarity matching |
| `/safety/status` | GET | Returns runtime resource consumption and remaining execution budget |
| `/health` | GET | Service health check |

---

## Canonical Repository Structure

```
researchforge_ecrm/
├── researchforge/
│   ├── pipeline/            # ResearchController, discovery orchestration
│   ├── policy/              # LinUCB bandit, transfer gate, policy learner
│   ├── discovery/           # OperatorGenome, DSLCompiler, MetaController
│   ├── transfer/            # UtilityLearner, CounterfactualArbitrator, sampling
│   ├── genome/              # ModelGenome, TargetModelGenome, operators
│   ├── memory/              # ECRM, posterior memory, adaptive trajectory
│   ├── vrdeg/               # VRDEG graph, nodes, edges, projector
│   ├── diagnosis/           # FailureTaxonomy rule-based diagnostic engine
│   ├── evaluators/          # Scikit-learn estimator evaluators
│   ├── safety/              # SafeRunner subprocess sandbox
│   ├── benchmarks/          # RDE-Bench tasks, continuity task suites
│   ├── api/                 # FastAPI server, DTOs, application services
│   └── config/              # RFConfig, EnvConfig
├── tests/                   # 66 test modules (unit, integration, contracts)
├── scripts/                 # Benchmarks B1, B2, B3, B6/B7, B9, B10, RF-3 demo
├── docs/                    # Integration plan, literature positioning, reports
├── RELEASE_MANIFEST.json    # Release manifest v1.0.0 (Phase RF-3)
├── CHANGELOG.md             # Formal release and change ledger
├── requirements.txt         # Core dependencies
└── pytest.ini               # Pytest configuration
```

---

## License

This reference implementation is licensed under the Apache License 2.0. See `LICENSE` for details.

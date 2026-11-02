"""ChemReact: A Reaction-Kinetics-Based Mathematical Paradigm for Predictive
Software Defect Detection and Test-Case Generation.

Mathematical Formulation:
-------------------------
1. Reactants and Physical State Variables:
   - Code Concentration [C]: Active code volume, token volume, Halstead volume.
   - Churn Flux [D]: Commit additions, modifications, and churn velocity.
   - Structural Complexity [K]: McCabe cyclomatic complexity, coupling, AST depth.
   - Effective Thermal Entropy (T_eff): Sprint stress, deadline compression,
     commit tempo variance, team cognitive entropy.
   - Activation Energy Barrier (E_a):
       E_a = E_0 + sum_i alpha_i * Q_i
     where Q_i represents defensive quality barriers:
       Q_1: Test coverage ratio (theta_cov in [0, 1])
       Q_2: Static typing rigor (theta_type in [0, 1])
       Q_3: Code review depth (theta_rev in [0, 1])
       Q_4: Architectural modularity and cohesion (theta_mod in [0, 1])

2. Defect Formation Rate Law (Arrhenius Kinetics):
   The forward reaction rate constant for defect formation is governed by:
       k_f = A * exp(-E_a / (R * T_eff))
   where A is the collision frequency factor and R is the software gas constant.
   The forward defect reaction flux is:
       r_f = k_f * [C]^alpha * [D]^beta * [K]^gamma

3. Reagent Neutralization and Test-Case Titration:
   Test execution is modeled as a reverse neutralization reaction:
       DefectSite_j + TestReagent_i -> NeutralizedSite_j (with affinity kappa_ij)
   where binding affinity is given by transition-state Gibbs free energy:
       kappa_ij = exp(-Delta G_ij / (k_B * T_eff))
   The equivalence point represents complete defect site neutralization.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .tasks import Task


@dataclass
class ChemReactKinetics:
    """Stores the physical kinetic parameters for a software module."""
    code_concentration_C: float      # [C]
    churn_flux_D: float              # [D]
    cyclomatic_complexity_K: float   # [K]
    effective_temperature_T: float   # T_eff
    activation_energy_Ea: float      # E_a
    arrhenius_rate_kf: float         # k_f = A * exp(-E_a / (R * T_eff))
    reaction_flux_rf: float          # r_f = k_f * C^alpha * D^beta
    gibbs_free_energy_dG: float      # Delta G^ddagger
    equilibrium_constant_Keq: float  # K_eq = k_f / k_r
    chemical_potential_grad: float   # nabla mu
    neutralization_affinity_idx: float  # kappa_bar
    reagent_diffusion_coeff: float   # D_diff


def generate_chemreact_dataset(
    seed: int = 42,
    n_samples: int = 600,
    noise_level: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates a synthetic software defect dataset derived strictly from
    Arrhenius reaction kinetics and chemical equilibrium dynamics.

    Returns:
        X: Feature matrix of shape (n_samples, 12) containing the kinetic variables.
        y: Binary ground-truth defect indicators (0 = clean, 1 = defect).
        feature_names: Array of feature name strings.
    """
    rng = np.random.RandomState(seed)

    # Physical constants for the software reaction vessel
    R_GAS = 1.0       # Normalized software gas constant
    A_FREQ = 12.5     # Pre-exponential collision frequency factor
    K_BOLTZMANN = 1.0 # Boltzmann equivalent constant
    E_0 = 1.8         # Baseline activation energy barrier

    X_list = []
    y_list = []

    for _ in range(n_samples):
        # 1. Primary state variables
        # Active code volume concentration [C] (normalized scale 0.5 to 10.0)
        C = rng.gamma(shape=2.5, scale=1.2) + 0.5
        # Churn feed flux [D] (normalized velocity 0.1 to 8.0)
        D = rng.exponential(scale=1.5) + 0.1
        # Cyclomatic structural complexity [K] (normalized 1.0 to 15.0)
        K = rng.lognormal(mean=1.2, sigma=0.5) + 0.5

        # 2. Thermodynamic environment
        # Sprint temperature T_eff (normalized 0.8 to 3.5; high = sprint deadline crunch)
        T_eff = rng.uniform(0.9, 3.2)

        # 3. Defensive quality barriers (each in [0, 1])
        q_test = rng.beta(a=2.0, b=2.0)
        q_type = rng.beta(a=3.0, b=1.5)
        q_review = rng.beta(a=1.8, b=2.2)
        q_mod = rng.beta(a=2.5, b=2.0)

        # Combined activation barrier E_a
        E_a = E_0 + (1.2 * q_test) + (0.9 * q_type) + (1.4 * q_review) + (1.0 * q_mod)

        # 4. Arrhenius kinetic rate constant
        # k_f = A * exp(-E_a / (R * T_eff))
        exponent = -E_a / (R_GAS * T_eff)
        k_f = A_FREQ * math.exp(exponent)

        # 5. Forward reaction flux (rate law: r_f = k_f * C^0.8 * D^0.6 * K^0.4)
        r_f = k_f * (C ** 0.8) * (D ** 0.6) * (K ** 0.4)

        # 6. Reverse quenching rate and chemical equilibrium constant
        k_r = 1.8 * (q_test ** 1.5) * (q_review + 0.2) + 0.1
        K_eq = k_f / max(1e-4, k_r)

        # 7. Transition-state Gibbs free energy Delta G^ddagger
        # Higher Delta G implies higher barrier to defect transition state
        dG = E_a - (T_eff * 0.45 * (q_mod + 0.1))

        # 8. Chemical potential gradient nabla mu across architectural boundaries
        mu_grad = (C * K) / (E_a * T_eff)

        # 9. Test reagent neutralization affinity index
        # kappa = exp(-Delta G_test / (k_B * T_eff))
        dG_test = max(0.2, 2.5 - (1.5 * q_test) - (0.8 * q_review))
        affinity_idx = math.exp(-dG_test / (K_BOLTZMANN * T_eff))

        # 10. Reagent diffusion coefficient (inter-module coupling accessibility)
        d_diff = 1.0 / (1.0 + (K / 4.0) + (C / 8.0))

        feature_vector = [
            C,               # 0: [C] Code volume concentration
            D,               # 1: [D] Churn feed flux
            K,               # 2: [K] Cyclomatic complexity
            T_eff,           # 3: T_eff Effective temperature
            E_a,             # 4: E_a Activation energy barrier
            k_f,             # 5: k_f Arrhenius kinetic rate
            r_f,             # 6: r_f Forward reaction flux
            dG,              # 7: Delta G Transition-state free energy
            K_eq,            # 8: K_eq Equilibrium constant
            mu_grad,         # 9: nabla mu Chemical potential gradient
            affinity_idx,    # 10: kappa Neutralization affinity index
            d_diff,          # 11: D_diff Reagent diffusion coefficient
        ]

        # 11. Ground-truth defect state from net kinetic equilibrium
        # Defect formation is thermodynamically favorable when r_f exceeds
        # quenching flux, or when chemical potential gradient is severe.
        logit = (
            1.8 * math.log(r_f + 1.0)
            + 1.4 * math.log(K_eq + 1.0)
            + 0.9 * mu_grad
            - 2.2 * affinity_idx
            - 1.5 * E_a
            + rng.normal(0.0, noise_level)
        )

        p_defect = 1.0 / (1.0 + math.exp(-logit))
        y = 1 if p_defect >= 0.50 else 0

        X_list.append(feature_vector)
        y_list.append(y)

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int64)

    feature_names = np.array([
        "code_concentration_C",
        "churn_flux_D",
        "cyclomatic_complexity_K",
        "effective_temperature_T",
        "activation_energy_Ea",
        "arrhenius_rate_kf",
        "reaction_flux_rf",
        "gibbs_free_energy_dG",
        "equilibrium_constant_Keq",
        "chemical_potential_grad",
        "neutralization_affinity_idx",
        "reagent_diffusion_coeff",
    ])

    return X, y, feature_names


def chemreact_defect_detection_task(
    seed: int = 42,
    n_samples: int = 600,
    n_train: int = 120,
    target_metric: float = 0.88,
) -> Task:
    """Creates an RDE-Bench Task for ChemReact Predictive Software Defect Detection.

    Uses a training split (n_train=120) with validation (240) and testing (240)
    splits, ensuring model architecture and hyperparameter search play a
    decisive role in achieving the target metric.
    """
    X, y, _ = generate_chemreact_dataset(seed=seed, n_samples=n_samples)

    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, train_size=n_train, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest
    )

    return Task(
        name="chemreact_defect_detection",
        description=(
            "Predictive software defect detection via reaction kinetics "
            f"(Arrhenius activation energy, churn flux, thermal entropy; {n_train}-sample train budget)"
        ),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        metric_fn=accuracy_score,
        target_metric=target_metric,
    )


def chemreact_titration_task(
    seed: int = 42,
    n_samples: int = 500,
    n_train: int = 100,
    target_metric: float = 0.85,
) -> Task:
    """Creates an RDE-Bench Task for ChemReact Test-Case Titration Equivalence.

    Predicts whether a generated test-reagent suite successfully achieves
    neutralization equivalence over the active defect reactive sites.
    """
    rng = np.random.RandomState(seed + 100)
    X_kinetics, _, _ = generate_chemreact_dataset(seed=seed, n_samples=n_samples)

    # Titration suite features: reagent volume [T], diversity entropy H(T),
    # coverage overlap, binding affinity mean
    titrant_vol = rng.uniform(5.0, 50.0, size=(n_samples, 1))
    entropy_reagents = rng.uniform(0.5, 4.0, size=(n_samples, 1))

    # Combine kinetics with titrant parameters
    X_titration = np.hstack([X_kinetics, titrant_vol, entropy_reagents])

    # Titration equivalence ground-truth
    # Neutralization is achieved if titrant volume * affinity exceeds reaction flux
    equivalence_score = (
        (titrant_vol.ravel() * 0.25)
        * X_titration[:, 10]  # affinity index
        * (entropy_reagents.ravel() / 2.0)
        - (X_titration[:, 6] * 0.4)  # reaction flux
        + rng.normal(0.0, 0.1, size=n_samples)
    )
    y_titration = (equivalence_score > 0.0).astype(np.int64)

    X_train, X_rest, y_train, y_rest = train_test_split(
        X_titration, y_titration, train_size=n_train, random_state=seed, stratify=y_titration
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest
    )

    return Task(
        name="chemreact_titration_equivalence",
        description=(
            "Automated test-case titration equivalence prediction "
            f"(chemical affinity neutralization and reagent dosage; {n_train}-sample train budget)"
        ),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        metric_fn=accuracy_score,
        target_metric=target_metric,
    )

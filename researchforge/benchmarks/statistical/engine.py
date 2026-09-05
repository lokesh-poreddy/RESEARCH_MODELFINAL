"""researchforge/benchmarks/statistical/engine.py — Statistical Analysis Engine.

Phase 12A Statistical Invariants:
1. Primary endpoint H1 receives NO multiplicity penalty against secondary hypotheses.
2. Secondary family (H2-H4) is adjusted via Holm-Bonferroni step-down.
3. Matching preserves experimental blocks on (task_id, seed, order_id).
4. Bootstrap is structure-preserving across matched blocks; unstructured bootstrap is rejected.
5. Primary inference combines paired contrast, blocked bootstrap 95% CI, and paired permutation test.
6. Parametric t-test and Wilcoxon are diagnostic/secondary with normality checks.
7. Evaluates H4 non-monotonicity via pre-registered contrasts C1 > 0 and C2 < 0.
8. Executes all 5 pre-registered sensitivity analyses.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import scipy.stats as stats

from .models import (
    HypothesisDecision,
    HypothesisEvaluationResult,
    HypothesisType,
    PairedContrastRecord,
    PairedTestResult,
    SensitivityResult,
    StatisticalArtifact,
    ValidationStatus,
)
from .plan import HypothesisSpec, StatisticalAnalysisPlan


class StatisticalAnalysisEngine:
    """Independent statistical inference engine operating on raw benchmark observations."""

    def __init__(self, plan: StatisticalAnalysisPlan, random_seed: int = 42) -> None:
        self.plan = plan
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

    # ── 1. Matched Contrast Construction ─────────────────────────────────────

    def build_paired_contrasts(
        self,
        observations: List[Dict[str, Any]],
        endpoint: str,
        test_condition: str,
        control_condition: str,
        regime_filter: Optional[str] = None,
    ) -> List[PairedContrastRecord]:
        """Constructs matched pairs across identical experimental blocks (task_id, seed, order_id)."""
        test_map: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
        control_map: Dict[Tuple[str, int, str], Dict[str, Any]] = {}

        for obs in observations:
            # Filter by regime if pre-registered
            if regime_filter and obs.get("regime") != regime_filter:
                continue

            key = (obs["task_id"], obs["seed"], obs["order_id"])
            cond = obs["condition"]

            if cond == test_condition:
                test_map[key] = obs
            elif cond == control_condition:
                control_map[key] = obs

        common_keys = sorted(set(test_map.keys()).intersection(set(control_map.keys())))
        contrasts: List[PairedContrastRecord] = []

        for key in common_keys:
            task_id, seed, order_id = key
            t_obs = test_map[key]
            c_obs = control_map[key]

            t_val = float(t_obs.get(endpoint, 0.0))
            c_val = float(c_obs.get(endpoint, 0.0))
            diff = t_val - c_val

            record = PairedContrastRecord(
                block_id=f"{task_id}_{seed}_{order_id}",
                task_id=task_id,
                seed=seed,
                order_id=order_id,
                regime=t_obs.get("regime", "UNKNOWN"),
                test_condition=test_condition,
                control_condition=control_condition,
                test_value=t_val,
                control_value=c_val,
                paired_difference=diff,
                prior_task_count=t_obs.get("prior_task_count", 0),
            )
            contrasts.append(record)

        return contrasts

    # ── 2. Structure-Preserving Block Bootstrap ──────────────────────────────

    def structure_preserving_bootstrap_ci(
        self,
        contrasts: Sequence[PairedContrastRecord],
        n_resamples: int = 2000,
        confidence_level: float = 0.95,
    ) -> Tuple[float, float]:
        """Resamples matched experimental blocks with replacement to estimate confidence interval.

        Invariant (Correction 1 & 4):
        - Never resamples test and control values independently.
        - Preserves matched experimental block boundaries.
        - Raises ValueError if contrasts list is empty or malformed.
        """
        if not contrasts:
            raise ValueError("Cannot perform structure-preserving bootstrap on empty contrasts.")

        diffs = np.array([c.paired_difference for c in contrasts], dtype=np.float64)
        n = len(diffs)
        if n == 1:
            return float(diffs[0]), float(diffs[0])

        boot_means = np.empty(n_resamples, dtype=np.float64)
        for i in range(n_resamples):
            sample_idx = self.rng.randint(0, n, size=n)
            boot_means[i] = np.mean(diffs[sample_idx])

        alpha = 1.0 - confidence_level
        lower_pct = 100.0 * (alpha / 2.0)
        upper_pct = 100.0 * (1.0 - alpha / 2.0)

        ci_lower = float(np.percentile(boot_means, lower_pct))
        ci_upper = float(np.percentile(boot_means, upper_pct))
        return ci_lower, ci_upper

    # ── 3. Paired Permutation / Randomization Test ───────────────────────────

    def paired_permutation_test(
        self,
        contrasts: Sequence[PairedContrastRecord],
        n_permutations: int = 5000,
        alternative: str = "two-sided",
    ) -> float:
        """Tests the sharp null hypothesis of zero paired treatment effect via sign-flipping randomization."""
        if not contrasts:
            return 1.0

        diffs = np.array([c.paired_difference for c in contrasts], dtype=np.float64)
        n = len(diffs)
        obs_mean = float(np.mean(diffs))

        # Under H0, test and control labels are exchangeable within each matched block
        signs = self.rng.choice([-1.0, 1.0], size=(n_permutations, n))
        perm_diffs = signs * diffs
        perm_means = np.mean(perm_diffs, axis=1)

        if alternative == "greater":
            count = np.sum(perm_means >= obs_mean)
        elif alternative == "less":
            count = np.sum(perm_means <= obs_mean)
        else:  # two-sided
            count = np.sum(np.abs(perm_means) >= np.abs(obs_mean))

        p_val = float((count + 1.0) / (n_permutations + 1.0))
        return p_val

    # ── 4. Comprehensive Paired Test Estimation ──────────────────────────────

    def compute_paired_test(
        self,
        contrasts: List[PairedContrastRecord],
        endpoint: str,
        comparison_name: str,
        test_condition: str,
        control_condition: str,
        regime_filter: Optional[str] = None,
        alternative: str = "two-sided",
    ) -> PairedTestResult:
        """Computes point estimates, blocked bootstrap CI, permutation test, and diagnostic parametric tests."""
        if not contrasts:
            return PairedTestResult(
                endpoint=endpoint,
                comparison_name=comparison_name,
                test_condition=test_condition,
                control_condition=control_condition,
                regime_filter=regime_filter,
                n_pairs=0,
                mean_difference=0.0,
                median_difference=0.0,
                std_difference=0.0,
                ci_lower_95=0.0,
                ci_upper_95=0.0,
                cohen_dz=0.0,
                rank_biserial_r=0.0,
                permutation_p_value=1.0,
                is_normal=False,
            )

        diffs = np.array([c.paired_difference for c in contrasts], dtype=np.float64)
        n = len(diffs)

        mean_d = float(np.mean(diffs))
        median_d = float(np.median(diffs))
        std_d = float(np.std(diffs, ddof=1)) if n > 1 else 0.0

        # Cohen's dz
        cohen_dz = float(mean_d / std_d) if std_d > 1e-12 else 0.0

        # Matched rank-biserial correlation
        test_vals = np.array([c.test_value for c in contrasts], dtype=np.float64)
        ctrl_vals = np.array([c.control_value for c in contrasts], dtype=np.float64)

        rank_biserial = 0.0
        wilcoxon_p = None
        if n >= 3 and not np.all(diffs == 0):
            try:
                res_w = stats.wilcoxon(test_vals, ctrl_vals, zero_method="wilcox")
                wilcoxon_p = float(res_w.pvalue)
                # Compute rank-biserial correlation from signed ranks
                abs_diffs = np.abs(diffs)
                non_zero = abs_diffs > 0
                if np.any(non_zero):
                    nz_diffs = diffs[non_zero]
                    ranks = stats.rankdata(np.abs(nz_diffs))
                    pos_ranks = np.sum(ranks[nz_diffs > 0])
                    neg_ranks = np.sum(ranks[nz_diffs < 0])
                    tot_ranks = pos_ranks + neg_ranks
                    if tot_ranks > 0:
                        rank_biserial = float((pos_ranks - neg_ranks) / tot_ranks)
            except Exception:
                wilcoxon_p = None

        # Paired Student's t-test (secondary / diagnostic)
        paired_t_p = None
        if n >= 2 and std_d > 1e-12:
            try:
                res_t = stats.ttest_rel(test_vals, ctrl_vals)
                paired_t_p = float(res_t.pvalue)
            except Exception:
                paired_t_p = None

        # Normality diagnostic via Shapiro-Wilk
        shapiro_p = None
        is_normal = False
        if 3 <= n <= 5000:
            try:
                shapiro_stat, shapiro_p_val = stats.shapiro(diffs)
                shapiro_p = float(shapiro_p_val)
                is_normal = bool(shapiro_p >= 0.05)
            except Exception:
                is_normal = False

        # Block-aware bootstrap 95% Confidence Interval
        ci_lower, ci_upper = self.structure_preserving_bootstrap_ci(
            contrasts,
            n_resamples=self.plan.bootstrap_resamples,
            confidence_level=self.plan.confidence_level,
        )

        # Primary inference: paired permutation test
        perm_p = self.paired_permutation_test(
            contrasts,
            n_permutations=self.plan.permutation_resamples,
            alternative=alternative,
        )

        return PairedTestResult(
            endpoint=endpoint,
            comparison_name=comparison_name,
            test_condition=test_condition,
            control_condition=control_condition,
            regime_filter=regime_filter,
            n_pairs=n,
            mean_difference=mean_d,
            median_difference=median_d,
            std_difference=std_d,
            ci_lower_95=ci_lower,
            ci_upper_95=ci_upper,
            cohen_dz=cohen_dz,
            rank_biserial_r=rank_biserial,
            permutation_p_value=perm_p,
            wilcoxon_p_value=wilcoxon_p,
            paired_t_p_value=paired_t_p,
            normality_shapiro_p=shapiro_p,
            is_normal=is_normal,
            bootstrap_resamples=self.plan.bootstrap_resamples,
            resampling_unit=self.plan.resampling_unit,
        )

    # ── 5. Multiplicity Correction (Holm-Bonferroni) ─────────────────────────

    @staticmethod
    def apply_holm_bonferroni(raw_p_values: Dict[str, float]) -> Dict[str, float]:
        """Applies Holm-Bonferroni step-down correction over secondary hypothesis family.

        Invariant (Correction 2):
        - H1 is never adjusted against H2-H4.
        - Secondary p-values are adjusted strictly in step-down rank order: p_adj = min(1, (m - k + 1) * p).
        """
        sorted_items = sorted(raw_p_values.items(), key=lambda item: item[1])
        m = len(sorted_items)
        adjusted_map: Dict[str, float] = {}

        cum_max = 0.0
        for k, (hyp_id, p_val) in enumerate(sorted_items):
            multiplier = m - k
            p_adj = min(1.0, multiplier * p_val)
            cum_max = max(cum_max, p_adj)
            adjusted_map[hyp_id] = cum_max

        return adjusted_map

    # ── 6. Full Hypothesis Evaluation Pipeline ───────────────────────────────

    def evaluate_all_hypotheses(
        self,
        observations: List[Dict[str, Any]],
    ) -> Tuple[List[HypothesisEvaluationResult], List[PairedTestResult]]:
        """Evaluates H1 (primary) and H2-H4 (secondary) against the pre-registered plan."""
        eval_results: List[HypothesisEvaluationResult] = []
        paired_tests: List[PairedTestResult] = []
        secondary_p_values: Dict[str, float] = {}

        # ── Hypothesis H1: Primary Confirmatory ───────────────────────────────
        h1_spec = self.plan.hypotheses["H1"]
        h1_contrasts = self.build_paired_contrasts(
            observations=observations,
            endpoint=h1_spec.endpoint,
            test_condition=h1_spec.test_condition,
            control_condition=h1_spec.control_condition,
            regime_filter=h1_spec.regime_filter,
        )
        h1_test = self.compute_paired_test(
            contrasts=h1_contrasts,
            endpoint=h1_spec.endpoint,
            comparison_name="H1: Continuous vs Cold Start (Same Family)",
            test_condition=h1_spec.test_condition,
            control_condition=h1_spec.control_condition,
            regime_filter=h1_spec.regime_filter,
            alternative="two-sided",
        )
        paired_tests.append(h1_test)

        # H1 Decision (Primary endpoint: NO multiplicity adjustment applied)
        h1_raw_p = h1_test.permutation_p_value
        h1_adj_p = h1_raw_p  # Invariant (Correction 2): alpha=0.05 without adjustment
        h1_decision = HypothesisDecision.FAIL_TO_REJECT
        if h1_adj_p < self.plan.alpha_primary and h1_test.mean_difference > 0:
            h1_decision = HypothesisDecision.REJECT_NULL
        elif h1_test.n_pairs < 2:
            h1_decision = HypothesisDecision.INCONCLUSIVE

        eval_results.append(HypothesisEvaluationResult(
            hypothesis_id="H1",
            hypothesis_type=HypothesisType.PRIMARY_CONFIRMATORY,
            description=h1_spec.description,
            endpoint=h1_spec.endpoint,
            raw_p_value=h1_raw_p,
            adjusted_p_value=h1_adj_p,
            multiplicity_method="none_primary",
            effect_size=h1_test.cohen_dz,
            effect_metric="cohen_dz",
            ci_95=(h1_test.ci_lower_95, h1_test.ci_upper_95),
            decision=h1_decision,
            contrast_details=h1_test.to_dict(),
        ))

        # ── Hypothesis H2: Cross-Family Negative Transfer ────────────────────
        h2_spec = self.plan.hypotheses["H2"]
        h2_contrasts = self.build_paired_contrasts(
            observations=observations,
            endpoint=h2_spec.endpoint,
            test_condition=h2_spec.test_condition,
            control_condition=h2_spec.control_condition,
            regime_filter=h2_spec.regime_filter,
        )
        h2_test = self.compute_paired_test(
            contrasts=h2_contrasts,
            endpoint=h2_spec.endpoint,
            comparison_name="H2: Cross-Family Negative Transfer",
            test_condition=h2_spec.test_condition,
            control_condition=h2_spec.control_condition,
            regime_filter=h2_spec.regime_filter,
            alternative="two-sided",
        )
        paired_tests.append(h2_test)
        secondary_p_values["H2"] = h2_test.permutation_p_value

        # ── Hypothesis H3: Adaptive Retrieval Safety ─────────────────────────
        h3_spec = self.plan.hypotheses["H3"]
        h3_contrasts = self.build_paired_contrasts(
            observations=observations,
            endpoint=h3_spec.endpoint,
            test_condition=h3_spec.test_condition,
            control_condition=h3_spec.control_condition,
            regime_filter=h3_spec.regime_filter,
        )
        h3_test = self.compute_paired_test(
            contrasts=h3_contrasts,
            endpoint=h3_spec.endpoint,
            comparison_name="H3: Adaptive Retrieval Safety vs Trajectory Memory",
            test_condition=h3_spec.test_condition,
            control_condition=h3_spec.control_condition,
            regime_filter=h3_spec.regime_filter,
            alternative="two-sided",
        )
        paired_tests.append(h3_test)
        secondary_p_values["H3"] = h3_test.permutation_p_value

        # ── Hypothesis H4: Non-Monotonic Experience Accumulation ──────────────
        # Invariant (Correction 3): Pre-registered contrast C1 = DQ1 - DQ0 and C2 = DQ2 - DQ1
        # H4 supported iff C1 > 0 and C2 < 0 under pre-specified sequence
        cont_obs = [o for o in observations if o["condition"] == "CONTINUOUS_EXPERIENCE"]
        h4_dq_by_horizon: Dict[int, List[float]] = {}
        for o in cont_obs:
            h4_dq_by_horizon.setdefault(o["prior_task_count"], []).append(float(o["decision_quality"]))

        dq_0 = statistics.mean(h4_dq_by_horizon.get(0, [0.0])) if h4_dq_by_horizon.get(0) else 0.0
        dq_1 = statistics.mean(h4_dq_by_horizon.get(1, [0.0])) if h4_dq_by_horizon.get(1) else 0.0
        dq_2 = statistics.mean(h4_dq_by_horizon.get(2, [0.0])) if h4_dq_by_horizon.get(2) else 0.0

        c1 = dq_1 - dq_0
        c2 = dq_2 - dq_1
        c_comp = dq_1 - (dq_0 + dq_2) / 2.0

        # Permutation test on composite non-monotonic contrast C_comp
        # Block-level contrast for each (seed, order_id)
        blocks: Dict[Tuple[int, str], Dict[int, float]] = {}
        for o in cont_obs:
            blocks.setdefault((o["seed"], o["order_id"]), {})[o["prior_task_count"]] = float(o["decision_quality"])

        comp_contrasts: List[float] = []
        for (s, ord_id), hor_map in blocks.items():
            if 0 in hor_map and 1 in hor_map and 2 in hor_map:
                comp_contrasts.append(hor_map[1] - (hor_map[0] + hor_map[2]) / 2.0)

        h4_raw_p = 1.0
        h4_ci_lower, h4_ci_upper = 0.0, 0.0
        if comp_contrasts:
            c_arr = np.array(comp_contrasts, dtype=np.float64)
            obs_comp = float(np.mean(c_arr))
            # Permutation under null that horizon order is exchangeable
            p_count = 0
            n_perm = self.plan.permutation_resamples
            for _ in range(n_perm):
                flips = self.rng.choice([-1.0, 1.0], size=len(c_arr))
                if np.mean(flips * c_arr) >= obs_comp:
                    p_count += 1
            h4_raw_p = float((p_count + 1.0) / (n_perm + 1.0))

            # Blocked bootstrap on composite contrast
            boot_c = [np.mean(self.rng.choice(c_arr, size=len(c_arr), replace=True)) for _ in range(self.plan.bootstrap_resamples)]
            h4_ci_lower = float(np.percentile(boot_c, 2.5))
            h4_ci_upper = float(np.percentile(boot_c, 97.5))

        secondary_p_values["H4"] = h4_raw_p

        # ── Apply Holm-Bonferroni Correction to Secondary Family (H2-H4) ──────
        adjusted_p_map = self.apply_holm_bonferroni(secondary_p_values)

        # H2 Result
        h2_adj_p = adjusted_p_map["H2"]
        h2_decision = HypothesisDecision.FAIL_TO_REJECT
        if h2_adj_p < self.plan.alpha_secondary_family and h2_test.mean_difference < 0:
            h2_decision = HypothesisDecision.REJECT_NULL
        elif h2_test.n_pairs < 2:
            h2_decision = HypothesisDecision.INCONCLUSIVE

        eval_results.append(HypothesisEvaluationResult(
            hypothesis_id="H2",
            hypothesis_type=HypothesisType.SECONDARY_EXPLORATORY,
            description=h2_spec.description,
            endpoint=h2_spec.endpoint,
            raw_p_value=h2_test.permutation_p_value,
            adjusted_p_value=h2_adj_p,
            multiplicity_method=self.plan.multiplicity_procedure_secondary,
            effect_size=h2_test.cohen_dz,
            effect_metric="cohen_dz",
            ci_95=(h2_test.ci_lower_95, h2_test.ci_upper_95),
            decision=h2_decision,
            contrast_details=h2_test.to_dict(),
        ))

        # H3 Result
        h3_adj_p = adjusted_p_map["H3"]
        h3_decision = HypothesisDecision.FAIL_TO_REJECT
        if h3_adj_p < self.plan.alpha_secondary_family and h3_test.mean_difference < 0:
            h3_decision = HypothesisDecision.REJECT_NULL
        elif h3_test.n_pairs < 2:
            h3_decision = HypothesisDecision.INCONCLUSIVE

        eval_results.append(HypothesisEvaluationResult(
            hypothesis_id="H3",
            hypothesis_type=HypothesisType.SECONDARY_EXPLORATORY,
            description=h3_spec.description,
            endpoint=h3_spec.endpoint,
            raw_p_value=h3_test.permutation_p_value,
            adjusted_p_value=h3_adj_p,
            multiplicity_method=self.plan.multiplicity_procedure_secondary,
            effect_size=h3_test.cohen_dz,
            effect_metric="cohen_dz",
            ci_95=(h3_test.ci_lower_95, h3_test.ci_upper_95),
            decision=h3_decision,
            contrast_details=h3_test.to_dict(),
        ))

        # H4 Result: Directional support requires C1 > 0 and C2 < 0
        h4_adj_p = adjusted_p_map["H4"]
        h4_decision = HypothesisDecision.FAIL_TO_REJECT
        if c1 > 0 and c2 < 0 and h4_adj_p < self.plan.alpha_secondary_family:
            h4_decision = HypothesisDecision.REJECT_NULL
        elif len(comp_contrasts) < 2:
            h4_decision = HypothesisDecision.INCONCLUSIVE

        eval_results.append(HypothesisEvaluationResult(
            hypothesis_id="H4",
            hypothesis_type=HypothesisType.SECONDARY_EXPLORATORY,
            description=self.plan.hypotheses["H4"].description,
            endpoint=self.plan.primary_endpoint,
            raw_p_value=h4_raw_p,
            adjusted_p_value=h4_adj_p,
            multiplicity_method=self.plan.multiplicity_procedure_secondary,
            effect_size=round(c_comp, 4),
            effect_metric="composite_non_monotonic_contrast",
            ci_95=(h4_ci_lower, h4_ci_upper),
            decision=h4_decision,
            contrast_details={
                "dq_0": round(dq_0, 4),
                "dq_1": round(dq_1, 4),
                "dq_2": round(dq_2, 4),
                "c1_dq1_minus_dq0": round(c1, 4),
                "c2_dq2_minus_dq1": round(c2, 4),
                "composite_contrast": round(c_comp, 4),
                "directional_criterion_met": bool(c1 > 0 and c2 < 0),
            },
        ))

        return eval_results, paired_tests

    # ── 7. Pre-Registered Sensitivity Analyses ────────────────────────────────

    def run_all_sensitivity_analyses(
        self,
        observations: List[Dict[str, Any]],
        transfer_events: Optional[List[Dict[str, Any]]] = None,
    ) -> List[SensitivityResult]:
        """Executes all 5 pre-registered sensitivity dimensions."""
        sens_results: List[SensitivityResult] = []

        # 1. Task Ordering Sensitivity
        for ord_id in ("forward", "reverse", "cross_first"):
            sub_obs = [o for o in observations if o["order_id"] == ord_id]
            contrasts = self.build_paired_contrasts(
                sub_obs,
                endpoint="decision_quality",
                test_condition="CONTINUOUS_EXPERIENCE",
                control_condition="COLD_START",
            )
            if contrasts:
                diffs = [c.paired_difference for c in contrasts]
                mean_d = float(np.mean(diffs))
                ci_l, ci_u = self.structure_preserving_bootstrap_ci(contrasts)
                stability = "STABLE" if (ci_l > 0 or ci_u < 0) else "NEUTRAL"
                sens_results.append(SensitivityResult(
                    dimension="task_ordering",
                    stratum=ord_id,
                    endpoint="decision_quality",
                    n_pairs=len(contrasts),
                    mean_difference=mean_d,
                    ci_lower_95=ci_l,
                    ci_upper_95=ci_u,
                    stability=stability,
                ))

        # 2. Seed Subset (Leave-One-Out) Sensitivity
        available_seeds = sorted(list(set(o["seed"] for o in observations)))
        for s in available_seeds:
            sub_obs = [o for o in observations if o["seed"] != s]
            contrasts = self.build_paired_contrasts(
                sub_obs,
                endpoint="decision_quality",
                test_condition="CONTINUOUS_EXPERIENCE",
                control_condition="COLD_START",
            )
            if contrasts:
                diffs = [c.paired_difference for c in contrasts]
                mean_d = float(np.mean(diffs))
                ci_l, ci_u = self.structure_preserving_bootstrap_ci(contrasts)
                stability = "STABLE" if (ci_l > 0 or ci_u < 0) else "NEUTRAL"
                sens_results.append(SensitivityResult(
                    dimension="seed_subset",
                    stratum=f"exclude_seed_{s}",
                    endpoint="decision_quality",
                    n_pairs=len(contrasts),
                    mean_difference=mean_d,
                    ci_lower_95=ci_l,
                    ci_upper_95=ci_u,
                    stability=stability,
                ))

        # 3. Task Regime Sensitivity
        for regime in ("SAME_FAMILY", "CROSS_FAMILY"):
            sub_obs = [o for o in observations if o.get("regime") == regime]
            contrasts = self.build_paired_contrasts(
                sub_obs,
                endpoint="decision_quality",
                test_condition="CONTINUOUS_EXPERIENCE",
                control_condition="COLD_START",
            )
            if contrasts:
                diffs = [c.paired_difference for c in contrasts]
                mean_d = float(np.mean(diffs))
                ci_l, ci_u = self.structure_preserving_bootstrap_ci(contrasts)
                stability = "STABLE" if (ci_l > 0 or ci_u < 0) else "NEUTRAL"
                sens_results.append(SensitivityResult(
                    dimension="task_regime",
                    stratum=regime,
                    endpoint="decision_quality",
                    n_pairs=len(contrasts),
                    mean_difference=mean_d,
                    ci_lower_95=ci_l,
                    ci_upper_95=ci_u,
                    stability=stability,
                ))

        # 4. Experience Horizon Sensitivity
        for h in (0, 1, 2):
            sub_obs = [o for o in observations if o["prior_task_count"] == h]
            contrasts = self.build_paired_contrasts(
                sub_obs,
                endpoint="decision_quality",
                test_condition="CONTINUOUS_EXPERIENCE",
                control_condition="COLD_START",
            )
            if contrasts:
                diffs = [c.paired_difference for c in contrasts]
                mean_d = float(np.mean(diffs))
                ci_l, ci_u = self.structure_preserving_bootstrap_ci(contrasts)
                stability = "STABLE" if (ci_l > 0 or ci_u < 0) else "NEUTRAL"
                sens_results.append(SensitivityResult(
                    dimension="experience_horizon",
                    stratum=f"horizon_{h}",
                    endpoint="decision_quality",
                    n_pairs=len(contrasts),
                    mean_difference=mean_d,
                    ci_lower_95=ci_l,
                    ci_upper_95=ci_u,
                    stability=stability,
                ))

        # 5. Transfer Epsilon Sensitivity
        if transfer_events:
            for eps in self.plan.transfer_epsilon_levels:
                # Re-classify transfer deltas under eps threshold
                pos = 0
                neg = 0
                for e in transfer_events:
                    ctx = e.get("context_level")
                    delta = e.get("delta")
                    if ctx is not None and ctx > 0 and delta is not None:
                        if delta > eps:
                            pos += 1
                        elif delta < -eps:
                            neg += 1
                tot = max(1, len(transfer_events))
                net_rate = (pos - neg) / tot
                sens_results.append(SensitivityResult(
                    dimension="transfer_epsilon",
                    stratum=f"eps_{eps}",
                    endpoint="net_transfer_rate",
                    n_pairs=tot,
                    mean_difference=net_rate,
                    ci_lower_95=net_rate,
                    ci_upper_95=net_rate,
                    stability="STABLE" if net_rate > 0 else "NEGATIVE",
                ))

        return sens_results

    # ── 8. Canonical Statistical Artifact Generation ─────────────────────────

    def generate_statistical_artifact(
        self,
        raw_benchmark_artifact_path: str,
        validation_status: ValidationStatus = ValidationStatus.METHOD_VALIDATION,
        output_path: Optional[str] = None,
    ) -> StatisticalArtifact:
        """Loads raw benchmark data, executes full SAP, and generates fingerprinted StatisticalArtifact."""
        with open(raw_benchmark_artifact_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        raw_benchmark_fp = raw_data.get("artifact_fingerprint", "")
        raw_observations = raw_data.get("raw_observations", [])
        raw_transfers = raw_data.get("transfer_classifications", [])

        # Missing data audit
        expected_count = len(raw_observations)
        valid_count = sum(1 for o in raw_observations if "decision_quality" in o and o["decision_quality"] is not None)
        missing_count = expected_count - valid_count
        missing_report = {
            "expected_observations": expected_count,
            "observed_observations": valid_count,
            "missing_observations": missing_count,
            "exclusion_rule": self.plan.exclusion_policy,
            "missingness_policy": self.plan.missing_data_policy,
            "exclusions_applied": 0,
        }

        # Evaluate hypotheses
        hyp_results, paired_tests = self.evaluate_all_hypotheses(raw_observations)

        # Run sensitivity analyses
        sens_results = self.run_all_sensitivity_analyses(raw_observations, raw_transfers)

        # Environment metadata
        software_metadata = {
            "python": "3.13",
            "numpy": np.__version__,
            "scipy": stats.__version__ if hasattr(stats, "__version__") else "1.15.2",
            "stat_engine_version": "1.0.0",
        }

        artifact = StatisticalArtifact(
            artifact_version="1.0.0",
            validation_status=validation_status.value,
            plan_version=self.plan.plan_version,
            plan_fingerprint=self.plan.plan_fingerprint,
            code_revision=self.plan.code_revision,
            raw_benchmark_artifact_path=raw_benchmark_artifact_path,
            raw_benchmark_fingerprint=raw_benchmark_fp,
            primary_endpoint=self.plan.primary_endpoint,
            secondary_endpoints=self.plan.secondary_endpoints,
            hypothesis_results=[h.to_dict() for h in hyp_results],
            paired_test_results=[p.to_dict() for p in paired_tests],
            sensitivity_results=[s.to_dict() for s in sens_results],
            missing_data_report=missing_report,
            statistical_software=software_metadata,
        )
        artifact.artifact_fingerprint = artifact.compute_fingerprint()

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(artifact.to_dict(), f, indent=2)

        return artifact

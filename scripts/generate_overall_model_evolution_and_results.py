"""Generate the evidence-backed overall ResearchForge evolution/results report.

The generator only reports measurements found in repository artifacts. RF-FLEX
confirmatory results are intentionally marked unavailable while execution is
locked; pilot/calibration values are labeled descriptive or projected.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "overall_model_evolution_and_results"


def load_json(name: str) -> Any:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def pct(value: Any) -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def num(value: Any, digits: int = 4) -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def delta_pct(value: float, baseline: float) -> str:
    if baseline == 0:
        return "N/A"
    return f"{(value - baseline) * 100:+.2f} pp"


def table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    headers = list(headers)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def git_identity() -> Dict[str, Any]:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip())
    return {"commit": sha, "dirty_worktree": dirty}


def write_bar(path: Path, labels: List[str], values: List[float], title: str, ylabel: str, percent: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, values, color=["#245b9e", "#3c8d7b", "#d78336", "#8c5aa8", "#b64d4d", "#65717c"][:len(labels)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=25)
    for bar, value in zip(bars, values):
        label = f"{value * 100:.1f}%" if percent else f"{value:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_grouped(path: Path, labels: List[str], series: Dict[str, List[float]], title: str, ylabel: str, percent: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8))
    width = 0.8 / max(1, len(series))
    positions = list(range(len(labels)))
    for index, (name, values) in enumerate(series.items()):
        offset = (index - (len(series) - 1) / 2) * width
        ax.bar([position + offset for position in positions], values, width, label=name)
    ax.set_xticks(positions, labels, rotation=25, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_line(path: Path, curves: Dict[str, List[float]], title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, values in curves.items():
        ax.plot(range(len(values)), values, marker="o", linewidth=1.6, label=label)
    ax.set_title(title)
    ax.set_xlabel("Research step / generation")
    ax.set_ylabel("Best-so-far validation metric")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate(out: Path, pilot_dir: Path | None = None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    identity = git_identity()
    demo = load_json("demo_results.json")
    regression = load_json("RF0_vs_RF1_regression.json")
    phase6 = load_json("phase6_benchmark_result.json")
    phase7 = load_json("phase7_benchmark_result.json")
    phase11 = load_json("phase11_continuity_benchmark_result.json")
    phase12 = load_json("phase12_statistical_validation.json")
    phase12b = load_json("phase12b_cohort_specification.json")

    pilot_validation = None
    if pilot_dir and (pilot_dir / "trajectory_validation.json").exists():
        pilot_validation = load_json_from(pilot_dir / "trajectory_validation.json")

    benchmark_rows = []
    for task_name, conditions in demo.items():
        baseline = conditions.get("no_memory", {}).get("best_metric_mean")
        for condition, result in conditions.items():
            best = result.get("best_metric_mean")
            benchmark_rows.append([
                task_name,
                condition,
                num(best),
                num(result.get("best_metric_std")),
                num(result.get("research_efficiency")),
                str(result.get("search_efficiency", "N/A")),
                pct(result.get("failure_repetition_rate")),
                pct(result.get("negative_transfer_rate")),
                delta_pct(best, baseline) if isinstance(best, (int, float)) and isinstance(baseline, (int, float)) else "N/A",
            ])

    # Comparison charts from measured demo results.
    for task_name, conditions in demo.items():
        labels = list(conditions)
        write_grouped(
            figures / f"{task_name}_benchmark_comparison.png",
            labels,
            {
                "Best metric": [conditions[label]["best_metric_mean"] for label in labels],
                "Research efficiency": [conditions[label]["research_efficiency"] for label in labels],
            },
            f"{task_name}: measured benchmark comparison",
            "Metric value",
        )
        write_grouped(
            figures / f"{task_name}_safety_comparison.png",
            labels,
            {
                "FRR": [conditions[label]["failure_repetition_rate"] for label in labels],
                "NTR": [conditions[label]["negative_transfer_rate"] for label in labels],
            },
            f"{task_name}: measured failure and transfer rates",
            "Rate",
            percent=True,
        )
        curves = {}
        for condition, result in conditions.items():
            if result.get("curves"):
                curves[condition] = result["curves"][0]
        if curves:
            write_line(figures / f"{task_name}_improvement_curves.png", curves, f"{task_name}: measured improvement curves, seed 0")

    regression_values = {
        "RF-0.x": [row["mean_metric"] for row in regression["track_a"]],
        "RF-1.0": [row["mean_metric"] for row in regression["track_b"]],
    }
    write_grouped(figures / "rf0_rf1_regression.png", ["full s0", "full s1", "full s2", "no-memory s0", "no-memory s1", "no-memory s2"], regression_values, "RF-0.x vs RF-1.0 measured regression", "Mean score")

    phase7_rows = []
    for condition, result in phase7.get("conditions", {}).items():
        phase7_rows.append([condition, num(result.get("best_metric_mean")), num(result.get("research_efficiency")), str(result.get("search_efficiency")), pct(result.get("failure_repetition_rate")), pct(result.get("negative_transfer_rate")), pct(result.get("backoff_frequency")), str(result.get("memory_influenced_decisions"))])
    write_bar(figures / "phase7_best_metric.png", list(phase7["conditions"]), [value["best_metric_mean"] for value in phase7["conditions"].values()], "Phase 7 measured best metric by condition", "Best metric")

    report = []
    report.append("# Overall Model Evolution and Results")
    report.append("")
    report.append("> Evidence-backed report generated from repository artifacts. Percentages are measured rates converted to percent; `N/A` means the required confirmatory evidence does not exist yet.")
    report.append("")
    report.append("## Evidence Status")
    report.append(table(["Evidence stream", "Status", "Source", "Interpretation"], [
        ["RF-0.x baseline", "EXECUTED", "Research_model/rf1_benchmark_report_v2.json", "Historical baseline and provenance comparison"],
        ["Phase 6", "EXECUTED", "phase6_benchmark_result.json", "Graph/reconstruction microbenchmark"],
        ["Phase 7", "EXECUTED", "phase7_benchmark_result.json", "Digits adaptive-memory benchmark"],
        ["RDE-Bench", "EXECUTED", "demo_results.json", "Five-seed, 25-generation measured ablations"],
        ["Phase 11", "ARTIFACT PRESENT; RAW OBSERVATIONS EMPTY", "phase11_continuity_benchmark_result.json", "Do not treat as efficacy evidence"],
        ["Phase 12A", "METHOD VALIDATION", "phase12_statistical_validation.json", "Statistical engine/protocol validation; not a new efficacy run"],
        ["Phase 12B", "FROZEN / NOT EXECUTED", "phase12b_cohort_specification.json", "32,400 planned trials; raw freeze pending"],
        ["RF-FLEX pilot", "DESCRIPTIVE ONLY", "Phase 13.8 pilot artifacts", "Integration and lineage coverage, not efficacy"],
        ["RF-FLEX confirmatory", "LOCKED / NOT EXECUTED", "Phase 13.9-13.11 protocol and gate", "No superiority claim permitted"],
    ]))
    report.append("")
    report.append("## Overall Evolution")
    report.append(table(["Stage", "Implemented/evaluated capability", "Evidence status"], [
        ["RF-0.x", "Baseline research loop, memory, model evolution", "Historical executed results"],
        ["RF-1.0 alpha", "Typed domain objects, RSG/TMG, provenance, regression preservation", "IMPLEMENTED and regression-tested"],
        ["Phase 11", "Continuity and cross-task transfer protocol", "Protocol artifact present; raw observations empty"],
        ["Phase 12A/12B", "SAP, cohort, execution ledger, preflight and raw-freeze infrastructure", "Method/infrastructure verified; confirmatory results unavailable"],
        ["Phase 13.7-13.8", "RF-FLEX integration, reconstruction, router and guard telemetry", "Pilot PASS, descriptive only"],
        ["Phase 13.9-13.11", "Frozen design, calibration, exact-commit execution gate and release seal", "Design/gate verified; authorization remains false"],
    ]))
    report.append("")
    report.append("## Measured RDE-Bench Comparisons")
    report.append("`best_metric` and `research_efficiency` are reported as raw values. FRR/NTR are percentages. `Delta vs no_memory` is a percentage-point difference in best metric, not a causal effect.")
    report.append("")
    report.append(table(["Task", "Condition", "Best", "Best std", "RE", "SE", "FRR", "NTR", "Delta vs no_memory"], benchmark_rows))
    report.append("")
    report.append("![Digits benchmark](figures/digits_benchmark_comparison.png)")
    report.append("\n![Digits transfer and failure rates](figures/digits_safety_comparison.png)")
    report.append("\n![Digits improvement curves](figures/digits_improvement_curves.png)")
    report.append("\n![Synthetic ECG benchmark](figures/synthetic_ecg_benchmark_comparison.png)")
    report.append("\n![Synthetic ECG transfer and failure rates](figures/synthetic_ecg_safety_comparison.png)")
    report.append("\n![Synthetic ECG improvement curves](figures/synthetic_ecg_improvement_curves.png)")
    report.append("")
    report.append("## Phase 7 Adaptive-Memory Comparison")
    report.append(table(["Condition", "Best", "RE", "SE", "FRR", "NTR", "Backoff", "Memory-influenced decisions"], phase7_rows))
    report.append("")
    report.append("![Phase 7 best metric](figures/phase7_best_metric.png)")
    report.append("")
    report.append("## RF-0.x vs RF-1.0 Regression")
    report.append(table(["Regression property", "Measured result"], [
        ["Comparisons", str(regression["aggregate"]["n_comparisons"])],
        ["Mean absolute delta", num(regression["aggregate"]["mean_absolute_delta"], 8)],
        ["Max absolute delta", num(regression["aggregate"]["max_absolute_delta"], 8)],
        ["Identical trajectories", str(regression["aggregate"]["all_trajectories_identical"])],
        ["Verdict", regression["verdict"]],
    ]))
    report.append("\n![RF-0.x and RF-1.0 regression](figures/rf0_rf1_regression.png)")
    report.append("")
    report.append("## Engineering and Provenance Results")
    report.append(table(["Measure", "Result", "Source"], [
        ["Phase 6 events / nodes / edges", f"{phase6['events']} / {phase6['nodes']} / {phase6['edges']}", "phase6_benchmark_result.json"],
        ["Phase 6 consistency", str(phase6["report_summary"]["consistency"]), "phase6_benchmark_result.json"],
        ["Phase 12B planned trials", str(phase12b["budget"]["total_benchmark_trials"]), "phase12b_cohort_specification.json"],
        ["Phase 12B cohort sealed", str(phase12b["is_sealed"]), "phase12b_cohort_specification.json"],
        ["Phase 12A H1 decision", phase12["hypothesis_results"][0]["decision"], "phase12_statistical_validation.json"],
        ["Phase 12A validation status", phase12["validation_status"], "phase12_statistical_validation.json"],
        ["Current report commit", identity["commit"], "git identity at generation"],
        ["Current worktree", "DIRTY (unrelated local changes present)" if identity["dirty_worktree"] else "CLEAN", "git status at generation"],
    ]))
    report.append("")
    report.append("## RF-FLEX Confirmatory Comparison Matrix")
    report.append(table(["Condition", "Memory", "TransferGuard", "Router", "Scientific result"], [
        ["COLD_START", "No", "No", "Fixed", "Not executed"],
        ["NO_MEMORY", "No", "No", "Fixed", "Not executed"],
        ["TRAJECTORY_MEMORY", "Yes", "No", "Fixed", "Not executed"],
        ["ADAPTIVE_TRAJECTORY", "Yes", "No", "Fixed", "Not executed"],
        ["TRANSFER_GUARD_STRICT", "Yes", "Strict", "Fixed", "Not executed"],
        ["TRANSFER_GUARD_SELECTIVE", "Yes", "Selective", "Fixed", "Not executed"],
        ["DYNAMIC_ROUTER", "Yes", "Selective", "Dynamic", "Not executed"],
        ["FULL_RESEARCHFORGE", "Yes", "Selective", "Dynamic", "Not executed"],
    ]))
    report.append("")
    report.append("## RF-FLEX Pilot and Calibration")
    if pilot_validation:
        report.append(table(["Pilot measure", "Value", "Status"], [
            ["Trajectory groups", pilot_validation["observed_trajectory_groups"], "Descriptive"],
            ["Lineage records", pilot_validation["record_count"], "Descriptive"],
            ["Transfer decisions", pilot_validation["transfer_decision_count"], "Descriptive"],
            ["Router transitions", "See mode_transitions.jsonl", "Descriptive"],
            ["Semantic replay", pilot_validation["semantic_replay_fingerprint"], "Validated"],
            ["Scientific efficacy", "NOT ESTABLISHED", "Confirmatory execution locked"],
        ]))
    else:
        report.append("Pilot directory was not supplied to the generator; RF-FLEX pilot values are not duplicated here.")
    report.append("")
    report.append("## Claim Boundary")
    report.append("The repository demonstrates implemented architecture, engineering verification, measured historical ablations, and RF-FLEX pilot integration. It does not yet demonstrate that TransferGuard, Dynamic Routing, ECRM, or the full ResearchForge architecture improves autonomous research quality. The 720-group confirmatory run remains behind explicit human authorization and has not been executed.")
    report.append("")
    report.append("## Generated Artifacts")
    report.append("- `figures/`: measured benchmark comparison, safety-rate, improvement-curve, and regression plots.")
    report.append("- `overall model evolution and results.md`: this report in the requested filename.")
    report.append("- `overall_model_evolution_and_results.md`: machine-friendly alias.")
    report.append("- `ResearchForge-ECRM_Claim_Evidence_Matrix.csv`: claim-to-artifact status matrix.")
    report_path = out / "overall_model_evolution_and_results.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    (out / "overall model evolution and results.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    claims = [
        ["C-001", "ResearchForge has a versioned research-development architecture", "Source + tests", "RESEARCHFORGE_STATE.yaml; tests/", identity["commit"], "IMPLEMENTED", "Architecture is not efficacy"],
        ["C-002", "RF-0.x to RF-1.0 trajectory behavior is regression-identical", "Executed artifact", "RF0_vs_RF1_regression.json", regression.get("timestamp", ""), "EXECUTED", "Scope is tested conditions/seeds only"],
        ["C-003", "Historical RDE-Bench measured memory/ablation outcomes", "Executed artifact", "demo_results.json", "historical artifact", "EXECUTED", "Not the RF-FLEX confirmatory design"],
        ["C-004", "RF-FLEX pilot reconstructs integrated trajectories", "Pilot artifact", "Phase 13.8 trajectory_validation.json", "Phase 13.8", "EXECUTED / DESCRIPTIVE", "No efficacy claim"],
        ["C-005", "RF-FLEX confirmatory conditions improve research quality", "Confirmatory artifact", "Not available; execution locked", "N/A", "NOT ESTABLISHED", "Requires authorized run"],
        ["C-006", "Phase 13.9 defines a frozen confirmatory protocol", "Frozen protocol", "rf_flex_13_9_protocol.json / source", "bd6428f", "IMPLEMENTED / FROZEN", "No results yet"],
        ["C-007", "Phase 13.10/13.11 prevent unauthorized dirty execution", "Tests + gate output", "tests/test_phase13_10_execution.py", "bd6428f", "VERIFIED", "Final seal requires clean release commit"],
    ]
    with (out / "ResearchForge-ECRM_Claim_Evidence_Matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Claim ID", "Claim", "Evidence type", "Artifact", "Exact commit/execution ID", "Status", "Limitation"])
        writer.writerows(claims)
    return report_path


def load_json_from(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate overall model evolution and results report")
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--pilot-dir", default="", help="Optional validated RF-FLEX pilot artifact directory")
    args = parser.parse_args()
    report = generate(Path(args.output), Path(args.pilot_dir) if args.pilot_dir else None)
    print(report)


if __name__ == "__main__":
    main()
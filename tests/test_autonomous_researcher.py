"""Unit tests for AutonomousResearchEngine and dynamic theme dilution."""

import pytest
from researchforge.engine.autonomous_researcher import (
    ThemeDilutionEngine,
    DynamicTaskGenerator,
    AutonomousResearchEngine,
)


def test_theme_dilution_equitest():
    topic = "EquiTest: A Chemical Equilibrium Model for Regression Stability and Software Change Impact Prediction"
    theme = ThemeDilutionEngine.dilute_topic(topic)
    assert "Equilibrium" in theme.core_title
    assert len(theme.state_variables) >= 4
    assert len(theme.equations) >= 2
    assert any("Q_r" in eq["latex"] for eq in theme.equations)


def test_dynamic_task_generator():
    topic = "EquiTest: A Chemical Equilibrium Model for Regression Stability and Software Change Impact Prediction"
    theme = ThemeDilutionEngine.dilute_topic(topic)
    task, X_all, y_all, feat_names = DynamicTaskGenerator.generate_task(theme, seed=42, n_samples=300, n_train=60)
    assert task.X_train.shape[0] == 60
    assert task.X_val.shape[0] == 120
    assert task.X_test.shape[0] == 120
    assert X_all.shape[0] == 300
    assert task.X_train.shape[1] == 12
    assert len(feat_names) == 12
    assert set(task.y_train).issubset({0, 1})


def test_autonomous_research_engine_execution():
    topic = "EquiTest: A Chemical Equilibrium Model for Regression Stability and Software Change Impact Prediction"
    engine = AutonomousResearchEngine(seed=42)
    res = engine.execute(
        topic=topic,
        n_generations=2,
        condition="contextual_policy",
        enable_rf3_discovery=True,
    )
    assert res["status"] == "success"
    assert res["topic"] == topic
    assert "kpi" in res
    assert res["kpi"]["validation_metric"] > 0.80
    assert res["kpi"]["frr"] == 0.0000
    assert "rf3_operator" in res
    assert res["rf3_operator"]["name"] == "EquiTestEquilibriumExpansion"
    assert len(res["report_markdown"]) > 1000
    assert "EquiTest" in res["report_markdown"]

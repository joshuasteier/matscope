"""
Tests for matscope core functionality.

These tests use synthetic data and don't require any
model backends — they test the probing, analysis, and
reporting logic in isolation.
"""

import numpy as np
import pytest


class TestLinearProbe:
    """Test linear probe classification and regression."""

    def test_classification_separable(self):
        """Linear probe should achieve high accuracy on linearly separable data."""
        from matscope.probes.linear import LinearProbe

        np.random.seed(42)
        n = 200
        X = np.vstack([
            np.random.randn(n // 2, 10) + 2,
            np.random.randn(n // 2, 10) - 2,
        ])
        y = np.array([0] * (n // 2) + [1] * (n // 2))

        probe = LinearProbe(task="classification")
        metrics = probe.fit_evaluate(X, y, cv_folds=3)

        assert "accuracy" in metrics
        assert metrics["accuracy"] > 0.9

    def test_regression(self):
        """Linear probe should fit a linear relationship."""
        from matscope.probes.linear import LinearProbe

        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 5)
        y = X @ np.array([1.0, -0.5, 0.3, 0.0, 0.8]) + np.random.randn(n) * 0.1

        probe = LinearProbe(task="regression")
        metrics = probe.fit_evaluate(X, y, cv_folds=3)

        assert "r2" in metrics
        assert metrics["r2"] > 0.8

    def test_selectivity_probe(self):
        """Selectivity probe should return additional metrics."""
        from matscope.probes.linear import SelectivityProbe

        np.random.seed(42)
        n = 200
        X = np.vstack([
            np.random.randn(n // 2, 10) + 2,
            np.random.randn(n // 2, 10) - 2,
        ])
        y = np.array([0] * (n // 2) + [1] * (n // 2))

        probe = SelectivityProbe(task="classification")
        metrics = probe.fit_evaluate(X, y, cv_folds=3)

        assert "selectivity" in metrics
        assert "effective_dim" in metrics


class TestMLPProbe:
    """Test MLP probe."""

    def test_nonlinear_classification(self):
        """MLP should handle nonlinearly separable data."""
        from matscope.probes.nonlinear import MLPProbe

        np.random.seed(42)
        n = 300
        X = np.random.randn(n, 2)
        y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int)

        probe = MLPProbe(task="classification", hidden_dim=32)
        metrics = probe.fit_evaluate(X, y, cv_folds=3)

        assert "accuracy" in metrics
        assert metrics["accuracy"] > 0.5  # Should do better than random on nonlinear

    def test_linear_nonlinear_gap(self):
        """Gap function should return valid metrics."""
        from matscope.probes.nonlinear import linear_nonlinear_gap

        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 5)
        y = (X[:, 0] > 0).astype(int)

        result = linear_nonlinear_gap(X, y, task="classification", cv_folds=3)

        assert "linear_accuracy" in result
        assert "mlp_accuracy" in result
        assert "gap" in result


class TestRepresentationSimilarity:
    """Test CKA and other similarity methods."""

    def test_cka_identical(self):
        """CKA of identical representations should be ~1."""
        from matscope.analysis.similarity import RepresentationSimilarity

        np.random.seed(42)
        X = np.random.randn(50, 10)
        reps = {"layer_0": X}

        analyzer = RepresentationSimilarity(method="cka")
        sim = analyzer.compute(reps, reps)

        assert sim.shape == (1, 1)
        assert abs(sim[0, 0] - 1.0) < 0.01

    def test_cka_orthogonal(self):
        """CKA of unrelated representations should be low."""
        from matscope.analysis.similarity import RepresentationSimilarity

        np.random.seed(42)
        reps_a = {"layer_0": np.random.randn(50, 10)}
        reps_b = {"layer_0": np.random.randn(50, 10)}

        analyzer = RepresentationSimilarity(method="cka")
        sim = analyzer.compute(reps_a, reps_b)

        assert sim[0, 0] < 0.5  # Should be low for random

    def test_cca(self):
        """CCA should work without errors."""
        from matscope.analysis.similarity import RepresentationSimilarity

        np.random.seed(42)
        reps = {"l0": np.random.randn(50, 10), "l1": np.random.randn(50, 8)}

        analyzer = RepresentationSimilarity(method="cca")
        sim = analyzer.pairwise_across_layers(reps)

        assert sim.shape == (2, 2)


class TestShiftAnalyzer:
    """Test distribution shift detection."""

    def test_mmd_no_shift(self):
        """MMD should be low for samples from same distribution."""
        from matscope.analysis.shift import ShiftAnalyzer

        np.random.seed(42)
        X = np.random.randn(100, 10)
        Y = np.random.randn(100, 10)

        analyzer = ShiftAnalyzer(method="mmd")
        score = analyzer.compute_shift(X, Y)

        assert score < 0.5  # Should be small

    def test_mmd_with_shift(self):
        """MMD should be high for shifted distributions."""
        from matscope.analysis.shift import ShiftAnalyzer

        np.random.seed(42)
        X = np.random.randn(100, 10)
        Y = np.random.randn(100, 10) + 5  # Large shift

        analyzer = ShiftAnalyzer(method="mmd")
        score = analyzer.compute_shift(X, Y)

        assert score > 0.1  # Should detect shift

    def test_fisher(self):
        """Fisher method should run."""
        from matscope.analysis.shift import ShiftAnalyzer

        np.random.seed(42)
        X = np.random.randn(100, 10)
        Y = np.random.randn(100, 10) + 2

        analyzer = ShiftAnalyzer(method="fisher")
        score = analyzer.compute_shift(X, Y)

        assert score > 0

    def test_cosine_drift(self):
        """Cosine drift should detect shifted means."""
        from matscope.analysis.shift import ShiftAnalyzer

        np.random.seed(42)
        X = np.random.randn(100, 10) + 1
        Y = np.random.randn(100, 10) - 1

        analyzer = ShiftAnalyzer(method="cosine_drift")
        score = analyzer.compute_shift(X, Y)

        assert score > 0


class TestLayerwiseAnalyzer:
    """Test layerwise representation statistics."""

    def test_effective_dimensionality(self):
        """Effective dim should be higher for diverse representations."""
        from matscope.analysis.layerwise import LayerwiseAnalyzer

        np.random.seed(42)
        X_diverse = np.random.randn(100, 20)
        X_collapsed = np.random.randn(100, 1) @ np.random.randn(1, 20)

        analyzer = LayerwiseAnalyzer()
        dim_diverse = analyzer.effective_dimensionality(X_diverse)
        dim_collapsed = analyzer.effective_dimensionality(X_collapsed)

        assert dim_diverse > dim_collapsed

    def test_analyze_all_layers(self):
        """Full analysis should return metrics for all layers."""
        from matscope.analysis.layerwise import LayerwiseAnalyzer

        np.random.seed(42)
        reps = {
            "layer_0": np.random.randn(100, 32),
            "layer_1": np.random.randn(100, 64),
            "layer_2": np.random.randn(100, 16),
        }
        labels = np.random.randint(0, 3, 100)

        analyzer = LayerwiseAnalyzer()
        results = analyzer.analyze_all_layers(reps, labels=labels)

        assert len(results) == 3
        for layer_name, metrics in results.items():
            assert "effective_dim" in metrics
            assert "isotropy" in metrics
            assert "entropy" in metrics
            assert "separability" in metrics


class TestProbeResult:
    """Test result containers."""

    def test_best_layer(self):
        from matscope.core import ProbeResult

        result = ProbeResult(
            probe_type="linear",
            target_property="test",
            layer_results={
                "layer_0": {"accuracy": 0.7, "accuracy_std": 0.05},
                "layer_1": {"accuracy": 0.9, "accuracy_std": 0.03},
                "layer_2": {"accuracy": 0.8, "accuracy_std": 0.04},
            },
        )

        assert result.best_layer == "layer_1"
        assert result.summary["accuracy"] == 0.9

    def test_to_dataframe(self):
        from matscope.core import ProbeResult

        result = ProbeResult(
            probe_type="linear",
            target_property="test",
            layer_results={
                "layer_0": {"accuracy": 0.7},
                "layer_1": {"accuracy": 0.9},
            },
        )

        df = result.to_dataframe()
        assert len(df) == 2
        assert "layer" in df.columns
        assert "accuracy" in df.columns

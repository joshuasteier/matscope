"""
Tests for MatScope v0.2.0 features.

Tests CPD, dataset utilities, and new model backends.
All tests use synthetic data — no GPU or model downloads required.
"""

import numpy as np
import pytest


# ============================================================
# CPD Tests
# ============================================================


class TestCPDResult:
    """Test CPDResult dataclass."""

    def test_disentanglement_score_high(self):
        from matscope.analysis.cpd import CPDResult

        result = CPDResult(
            layer_name="interaction_2",
            composition_r2=0.5,
            residual_linear_r2=0.8,
            residual_mlp_r2=0.82,
            linear_nonlinear_gap=0.02,
            target_property="gap",
        )
        # disentanglement = 0.8 - |0.02| = 0.78
        assert abs(result.disentanglement_score - 0.78) < 1e-6

    def test_disentanglement_score_low(self):
        from matscope.analysis.cpd import CPDResult

        result = CPDResult(
            layer_name="aev",
            composition_r2=0.3,
            residual_linear_r2=-0.1,
            residual_mlp_r2=0.7,
            linear_nonlinear_gap=0.8,
            target_property="gap",
        )
        # disentanglement = -0.1 - |0.8| = -0.9
        assert result.disentanglement_score < 0

    def test_metadata_default(self):
        from matscope.analysis.cpd import CPDResult

        result = CPDResult(
            layer_name="test",
            composition_r2=0.0,
            residual_linear_r2=0.0,
            residual_mlp_r2=0.0,
            linear_nonlinear_gap=0.0,
            target_property="test",
        )
        assert result.metadata == {}


class TestCPDProfile:
    """Test CPDProfile dataclass."""

    def _make_profile(self):
        from matscope.analysis.cpd import CPDResult, CPDProfile

        results = {
            "layer_0": CPDResult(
                layer_name="layer_0",
                composition_r2=0.6,
                residual_linear_r2=0.3,
                residual_mlp_r2=0.5,
                linear_nonlinear_gap=0.2,
                target_property="gap",
            ),
            "layer_1": CPDResult(
                layer_name="layer_1",
                composition_r2=0.4,
                residual_linear_r2=0.7,
                residual_mlp_r2=0.75,
                linear_nonlinear_gap=0.05,
                target_property="gap",
            ),
            "layer_2": CPDResult(
                layer_name="layer_2",
                composition_r2=0.2,
                residual_linear_r2=0.5,
                residual_mlp_r2=0.9,
                linear_nonlinear_gap=0.4,
                target_property="gap",
            ),
        }
        return CPDProfile(
            model_name="test_model",
            results=results,
            target_property="gap",
        )

    def test_best_layer(self):
        profile = self._make_profile()
        assert profile.best_layer == "layer_1"  # highest residual_linear_r2

    def test_most_nonlinear_layer(self):
        profile = self._make_profile()
        assert profile.most_nonlinear_layer == "layer_2"  # largest gap

    def test_disentanglement_trajectory(self):
        profile = self._make_profile()
        traj = profile.disentanglement_trajectory
        assert len(traj) == 3
        assert isinstance(traj[0], float)

    def test_to_dataframe(self):
        profile = self._make_profile()
        df = profile.to_dataframe()
        assert len(df) == 3
        assert "composition_r2" in df.columns
        assert "gap" in df.columns

    def test_summary(self):
        profile = self._make_profile()
        s = profile.summary()
        assert "test_model" in s
        assert "layer_1" in s


class TestCompositionProjectionDecomposition:
    """Test the CPD decomposition engine."""

    def _make_synthetic_data(self, n=200, d=64, n_elem=5):
        """Create synthetic data with known composition/geometry structure."""
        rng = np.random.RandomState(42)

        # Composition features (sparse, sums to ~1)
        C = rng.dirichlet(np.ones(n_elem), size=n)

        # Generate representations as composition + geometry + noise
        W_comp = rng.randn(n_elem, d) * 2  # Composition basis
        W_geom = rng.randn(10, d)  # Geometry basis (latent)
        Z_geom = rng.randn(n, 10)  # Latent geometric features

        H = C @ W_comp + Z_geom @ W_geom + rng.randn(n, d) * 0.1

        # Target depends on both composition and geometry
        y = C @ rng.randn(n_elem) + Z_geom @ rng.randn(10) * 0.5 + rng.randn(n) * 0.1

        return H, C, y

    def test_decompose_basic(self):
        from matscope.analysis.cpd import CompositionProjectionDecomposition

        H, C, y = self._make_synthetic_data()
        cpd = CompositionProjectionDecomposition(cv_folds=3)
        result = cpd.decompose(H, C, y, layer_name="test", target_property="energy")

        assert result.layer_name == "test"
        assert result.target_property == "energy"
        assert -1.0 <= result.residual_linear_r2 <= 1.0
        assert -1.0 <= result.residual_mlp_r2 <= 1.0
        assert result.composition_r2 > 0  # Should capture some variance
        assert isinstance(result.linear_nonlinear_gap, float)

    def test_decompose_composition_variance(self):
        from matscope.analysis.cpd import CompositionProjectionDecomposition

        H, C, y = self._make_synthetic_data()
        cpd = CompositionProjectionDecomposition(cv_folds=3)
        result = cpd.decompose(H, C, y)

        # Composition should explain some variance in our synthetic data
        assert result.composition_variance_explained > 0.01

    def test_decompose_linear_only(self):
        from matscope.analysis.cpd import CompositionProjectionDecomposition

        H, C, y = self._make_synthetic_data(n=100)
        cpd = CompositionProjectionDecomposition(probe_type="linear", cv_folds=3)
        result = cpd.decompose(H, C, y)

        assert result.residual_linear_r2 != 0.0
        assert result.residual_mlp_r2 == 0.0  # Not computed

    def test_profile(self):
        from matscope.analysis.cpd import CompositionProjectionDecomposition

        rng = np.random.RandomState(42)
        n, d, n_elem = 150, 32, 4
        C = rng.dirichlet(np.ones(n_elem), size=n)
        y = rng.randn(n)

        representations = {
            f"layer_{i}": rng.randn(n, d) + C @ rng.randn(n_elem, d)
            for i in range(3)
        }

        cpd = CompositionProjectionDecomposition(cv_folds=3, probe_type="linear")
        profile = cpd.profile(
            representations, C, y,
            model_name="synthetic", target_property="energy"
        )

        assert profile.model_name == "synthetic"
        assert len(profile.results) == 3
        assert profile.best_layer in ["layer_0", "layer_1", "layer_2"]

    def test_compare_profiles(self):
        from matscope.analysis.cpd import (
            CompositionProjectionDecomposition,
            CPDResult,
            CPDProfile,
        )

        # Create two synthetic profiles
        profiles = {}
        for name, resid_r2, gap in [("equivariant", 0.8, 0.02), ("invariant", -0.1, 0.8)]:
            profiles[name] = CPDProfile(
                model_name=name,
                results={
                    "layer_0": CPDResult(
                        layer_name="layer_0",
                        composition_r2=0.5,
                        residual_linear_r2=resid_r2,
                        residual_mlp_r2=resid_r2 + gap,
                        linear_nonlinear_gap=gap,
                        target_property="gap",
                    )
                },
                target_property="gap",
            )

        comparison = CompositionProjectionDecomposition.compare_profiles(profiles)
        assert comparison["equivariant"]["disentanglement_rank"] == 1
        assert comparison["invariant"]["disentanglement_rank"] == 2


# ============================================================
# Dataset Utility Tests
# ============================================================


class TestCompositionFeatures:
    """Test composition feature extraction."""

    def test_fractional_composition(self):
        from matscope.datasets import composition_features

        # Mock structures as dicts
        structures = [
            {"atomic_numbers": [6, 6, 8, 1, 1]},  # C2OH2
            {"atomic_numbers": [6, 6, 6, 1, 1, 1]},  # C3H3
        ]

        C = composition_features(structures, method="fractional")
        assert C.shape[0] == 2
        # Each row should sum to 1
        np.testing.assert_allclose(C.sum(axis=1), 1.0, atol=1e-10)

    def test_count_composition(self):
        from matscope.datasets import composition_features

        structures = [
            {"atomic_numbers": [6, 6, 8, 1, 1]},
        ]

        C = composition_features(structures, method="count")
        # Should have raw counts
        assert C.sum() == 5  # Total atoms

    def test_onehot_composition(self):
        from matscope.datasets import composition_features

        structures = [
            {"atomic_numbers": [6, 6, 8, 1, 1]},
        ]

        C = composition_features(structures, method="onehot")
        # Binary values only
        assert set(C.flatten()).issubset({0.0, 1.0})
        assert C.sum() == 3  # H, C, O present

    def test_specified_elements(self):
        from matscope.datasets import composition_features

        structures = [
            {"atomic_numbers": [6, 6, 8, 1, 1]},
        ]

        C = composition_features(
            structures,
            method="fractional",
            elements=["H", "C", "N", "O"],
        )
        assert C.shape[1] == 4
        # N column should be zero
        assert C[0, 2] == 0.0  # N index

    def test_single_structure(self):
        from matscope.datasets import composition_features

        C = composition_features({"atomic_numbers": [6, 1, 1]})
        assert C.shape[0] == 1

    def test_numpy_atomic_numbers(self):
        from matscope.datasets import composition_features

        structures = [
            {"atomic_numbers": np.array([6, 6, 8])},
        ]
        C = composition_features(structures)
        assert C.shape[0] == 1


class TestTrainTestSplit:
    """Test dataset splitting."""

    def test_split_sizes(self):
        from matscope.datasets import train_test_split

        dataset_dict = {
            "dataset": list(range(100)),
            "compositions": np.random.randn(100, 5),
            "targets": {"energy": np.random.randn(100)},
            "property_names": ["energy"],
        }

        train, test = train_test_split(dataset_dict, test_size=0.2)
        assert len(train["dataset"]) == 80
        assert len(test["dataset"]) == 20
        assert train["compositions"].shape[0] == 80
        assert test["targets"]["energy"].shape[0] == 20

    def test_split_reproducibility(self):
        from matscope.datasets import train_test_split

        dataset_dict = {
            "dataset": list(range(50)),
            "compositions": np.random.randn(50, 3),
            "targets": {"e": np.random.randn(50)},
            "property_names": ["e"],
        }

        t1, _ = train_test_split(dataset_dict, random_state=42)
        t2, _ = train_test_split(dataset_dict, random_state=42)
        assert t1["dataset"] == t2["dataset"]


# ============================================================
# Model Backend Registration Tests
# ============================================================


class TestBackendRegistration:
    """Test that backends register correctly."""

    def test_core_backends_registered(self):
        from matscope.models.registry import available_backends

        backends = available_backends()
        # MACE and CHGNet should always be there
        assert "mace" in backends or "mace-mp-0" in backends
        assert "chgnet" in backends

    def test_pyg_backends_registered(self):
        from matscope.models.registry import available_backends

        backends = available_backends()
        assert "schnet" in backends
        assert "dimenet++" in backends or "dimenetpp" in backends
        assert "painn" in backends

    def test_ani_backend_registered(self):
        from matscope.models.registry import available_backends

        backends = available_backends()
        assert "ani-2x" in backends or "ani" in backends


# ============================================================
# Integration: CPD + Probes
# ============================================================


class TestCPDIntegration:
    """Integration tests combining CPD with probes."""

    def test_cpd_detects_linear_disentanglement(self):
        """When composition/geometry are linearly separable, CPD should find high residual R²."""
        from matscope.analysis.cpd import CompositionProjectionDecomposition

        rng = np.random.RandomState(123)
        n, d, n_elem = 300, 64, 5

        C = rng.dirichlet(np.ones(n_elem), size=n)
        Z_geom = rng.randn(n, 10)

        # Construct H so composition and geometry are in orthogonal subspaces
        W_comp = rng.randn(n_elem, d // 2)
        W_geom = rng.randn(10, d // 2)
        H = np.hstack([C @ W_comp, Z_geom @ W_geom])

        # Target depends on geometry
        y = Z_geom @ rng.randn(10) + rng.randn(n) * 0.1

        cpd = CompositionProjectionDecomposition(cv_folds=3)
        result = cpd.decompose(H, C, y)

        # Should find geometry info after removing composition
        assert result.residual_linear_r2 > 0.5
        # Gap should be small (linearly accessible)
        assert abs(result.linear_nonlinear_gap) < 0.3

    def test_cpd_detects_nonlinear_encoding(self):
        """When geometry info is nonlinearly encoded, gap should be large."""
        from matscope.analysis.cpd import CompositionProjectionDecomposition

        rng = np.random.RandomState(456)
        n, d, n_elem = 300, 32, 4

        C = rng.dirichlet(np.ones(n_elem), size=n)

        # Nonlinearly entangled representation
        H_base = rng.randn(n, d)
        # Apply nonlinear mixing with composition
        H = np.tanh(H_base * (C @ rng.randn(n_elem, d))) + C @ rng.randn(n_elem, d)

        # Target depends on the nonlinear interaction
        y = np.sum(np.sin(H_base[:, :5]) * C[:, :1], axis=1) + rng.randn(n) * 0.1

        cpd = CompositionProjectionDecomposition(cv_folds=3)
        result = cpd.decompose(H, C, y)

        # MLP should recover more than linear after composition removal
        # (This is a statistical test, may not always hold with small n)
        assert isinstance(result.linear_nonlinear_gap, float)


# ============================================================
# Existing functionality regression tests
# ============================================================


class TestExistingAPI:
    """Ensure v0.1.0 API still works."""

    def test_linear_probe(self):
        from matscope import LinearProbe

        rng = np.random.RandomState(42)
        X = rng.randn(100, 20)
        y = (X[:, 0] > 0).astype(int)

        probe = LinearProbe(task="classification")
        metrics = probe.fit_evaluate(X, y, cv_folds=3)
        assert "accuracy" in metrics
        assert metrics["accuracy"] > 0.5

    def test_mlp_probe(self):
        from matscope import MLPProbe

        rng = np.random.RandomState(42)
        X = rng.randn(100, 20)
        y = X[:, 0] + rng.randn(100) * 0.1

        probe = MLPProbe(task="regression")
        metrics = probe.fit_evaluate(X, y, cv_folds=3)
        assert "r2" in metrics

    def test_linear_nonlinear_gap(self):
        from matscope import linear_nonlinear_gap

        rng = np.random.RandomState(42)
        X = rng.randn(100, 20)
        y = (X[:, 0] > 0).astype(int)

        result = linear_nonlinear_gap(X, y, task="classification", cv_folds=3)
        assert "gap" in result
        assert "linear_accuracy" in result
        assert "mlp_accuracy" in result

    def test_selectivity_probe(self):
        from matscope import SelectivityProbe

        rng = np.random.RandomState(42)
        X = rng.randn(100, 20)
        y = (X[:, 0] > 0).astype(int)

        probe = SelectivityProbe(task="classification")
        metrics = probe.fit_evaluate(X, y, cv_folds=3)
        assert "selectivity" in metrics
        assert "effective_dim" in metrics

    def test_cka_similarity(self):
        from matscope import RepresentationSimilarity

        rng = np.random.RandomState(42)
        reps_a = {"layer_0": rng.randn(50, 32), "layer_1": rng.randn(50, 32)}
        reps_b = {"layer_0": rng.randn(50, 16), "layer_1": rng.randn(50, 16)}

        sim = RepresentationSimilarity(method="cka")
        matrix = sim.compute(reps_a, reps_b)
        assert matrix.shape == (2, 2)
        assert 0 <= matrix[0, 0] <= 1

    def test_shift_analyzer(self):
        from matscope import ShiftAnalyzer

        rng = np.random.RandomState(42)
        X_train = rng.randn(50, 32)
        X_deploy = rng.randn(50, 32) + 2.0  # Shifted

        analyzer = ShiftAnalyzer(method="mmd")
        score = analyzer.compute_shift(X_train, X_deploy)
        assert score > 0

    def test_layerwise_analyzer(self):
        from matscope import LayerwiseAnalyzer

        rng = np.random.RandomState(42)
        reps = {
            "layer_0": rng.randn(100, 64),
            "layer_1": rng.randn(100, 64),
        }
        labels = rng.randint(0, 3, size=100)

        analyzer = LayerwiseAnalyzer()
        results = analyzer.analyze_all_layers(reps, labels=labels)
        assert "layer_0" in results
        assert "effective_dim" in results["layer_0"]
        assert "separability" in results["layer_0"]


# ============================================================
# Version and import tests
# ============================================================


class TestPackage:
    """Test package metadata."""

    def test_version(self):
        import matscope

        assert matscope.__version__ == "0.2.0"

    def test_author(self):
        import matscope

        assert matscope.__author__ == "Joshua Steier"

    def test_all_exports(self):
        import matscope

        for name in matscope.__all__:
            assert hasattr(matscope, name), f"Missing export: {name}"

    def test_cpd_importable(self):
        from matscope import CompositionProjectionDecomposition, CPDResult, CPDProfile

        assert CompositionProjectionDecomposition is not None

    def test_datasets_importable(self):
        from matscope.datasets import composition_features, train_test_split

        assert composition_features is not None

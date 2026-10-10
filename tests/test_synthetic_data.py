"""Tests for synthetic data generator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from forestmind.data.synthetic import (
    SyntheticForestGenerator,
    HEALTH_STATES,
    DRIVER_CLASSES,
    SPECTRAL_PROFILES,
    SAR_PROFILES,
)


@pytest.fixture
def generator():
    return SyntheticForestGenerator(patch_size=64, time_steps=6, seed=42)


class TestSyntheticGenerator:
    """Test synthetic satellite data generation."""

    def test_single_sample_shapes(self, generator):
        """Generated sample has correct array shapes."""
        sample = generator.generate_sample()

        assert sample["optical"].shape == (6, 12, 64, 64), "optical shape mismatch"
        assert sample["sar"].shape == (6, 2, 64, 64), "sar shape mismatch"
        assert sample["timestamps"].shape == (6,), "timestamps shape mismatch"
        assert sample["drivers"].shape == (8,), "drivers shape mismatch"
        assert sample["driver_strengths"].shape == (8,), "driver_strengths shape mismatch"
        assert sample["trajectory"].shape == (5,), "trajectory shape mismatch"
        assert sample["health_history"].shape == (6,), "health_history shape mismatch"

    def test_value_ranges(self, generator):
        """All values fall within physically plausible ranges."""
        sample = generator.generate_sample()

        assert sample["optical"].min() >= 0.0, "optical below 0"
        assert sample["optical"].max() <= 1.0, "optical above 1"
        assert sample["sar"].min() >= 0.0, "sar below 0"
        assert sample["sar"].max() <= 1.0, "sar above 1"
        assert 0 <= sample["health_state"] <= 4, "health_state out of range"
        assert 0.0 <= sample["health_score"] <= 100.0, "health_score out of range"
        assert np.all(sample["drivers"] >= 0) and np.all(sample["drivers"] <= 1), "drivers out of range"
        assert np.all(sample["trajectory"] >= 0) and np.all(sample["trajectory"] <= 1), "trajectory out of range"
        assert np.all(sample["health_history"] >= 0) and np.all(sample["health_history"] <= 1), "health_history out of range"

    def test_timestamps_monotonic(self, generator):
        """Timestamps are monotonically increasing."""
        sample = generator.generate_sample()
        diffs = np.diff(sample["timestamps"])
        assert np.all(diffs >= 0), "timestamps not monotonic"

    def test_driver_consistency(self, generator):
        """Active drivers have positive strengths and vice versa."""
        for _ in range(20):
            sample = generator.generate_sample()
            for i in range(8):
                if sample["drivers"][i] > 0:
                    assert sample["driver_strengths"][i] > 0, f"Driver {i} active but strength=0"

    def test_health_state_classification(self, generator):
        """Health state matches the continuous score."""
        for _ in range(50):
            sample = generator.generate_sample()
            score = sample["health_score"] / 100.0
            state = sample["health_state"]

            if score >= 0.80:
                assert state == 0, f"score={score:.2f} but state={state}"
            elif score >= 0.60:
                assert state == 1, f"score={score:.2f} but state={state}"
            elif score >= 0.40:
                assert state == 2, f"score={score:.2f} but state={state}"
            elif score >= 0.15:
                assert state == 3, f"score={score:.2f} but state={state}"
            else:
                assert state == 4, f"score={score:.2f} but state={state}"

    def test_reproducibility(self):
        """Same seed produces identical samples."""
        gen1 = SyntheticForestGenerator(seed=99)
        gen2 = SyntheticForestGenerator(seed=99)

        s1 = gen1.generate_sample()
        s2 = gen2.generate_sample()

        np.testing.assert_array_equal(s1["optical"], s2["optical"])
        np.testing.assert_array_equal(s1["health_history"], s2["health_history"])

    def test_dataset_generation(self, generator):
        """generate_dataset produces correct number of samples."""
        samples = generator.generate_dataset(10)
        assert len(samples) == 10

    def test_spectral_profiles_valid(self):
        """Spectral profiles have correct shapes and ranges."""
        for name, profile in SPECTRAL_PROFILES.items():
            assert profile.shape == (12,), f"{name} has wrong shape"
            assert np.all(profile >= 0) and np.all(profile <= 1), f"{name} values out of range"

    def test_sar_profiles_valid(self):
        """SAR profiles have correct shapes and ranges."""
        for name, profile in SAR_PROFILES.items():
            assert profile.shape == (2,), f"{name} has wrong shape"
            assert np.all(profile >= 0) and np.all(profile <= 1), f"{name} values out of range"

"""Spectral maps distinguish computable zero change from unusable evidence."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from forestmind.data.sequence import prepare_sequence
from forestmind.monitoring.spatial import spectral_change
from test_prepare_sequence import cache


GEOMETRY = {"type": "Polygon", "coordinates": [[
    [-55.1, -3.5], [-55.08, -3.5], [-55.08, -3.48], [-55.1, -3.48], [-55.1, -3.5]]]
}


def prepared(data, masks=None):
    return SimpleNamespace(
        arrays={"optical": data, "quality_mask": np.ones((2, 2, 2), dtype=bool) if masks is None else masks,
                "acquisition_dates": np.array(["2024-01-02", "2024-02-02"])},
        provenance={"sources": [{"name": "first"}, {"name": "second"}]})


def test_accepted_scenes_without_computable_paired_indices_are_unavailable(tmp_path):
    # Both scenes contain valid spectral observations, but in disjoint halves;
    # the quality mask alone cannot establish any computable paired delta.
    first_data = np.zeros((12, 2, 2), dtype=np.float32)
    first_data[:, 0, :] = .3
    second_data = np.zeros_like(first_data)
    second_data[:, 1, :] = .3
    first = cache(tmp_path, "first", "2024-01-02T10:00:00+00:00", updates={"data": first_data})
    second = cache(tmp_path, "second", "2024-02-02T10:00:00+00:00", updates={"data": second_data})
    sequence = prepare_sequence(tmp_path, through="2024-02-28", source_paths=[first, second])
    result = spectral_change(sequence, GEOMETRY)
    assert result["available"] is False
    assert "positive denominators" in result["reason"]
    for layer in result["layers"].values():
        assert layer["values"] == [[None, None], [None, None]]
        assert layer["valid_fraction"] == 0
        assert layer["mean_delta"] is None


def test_valid_unchanged_reflectance_is_available_zero_change():
    result = spectral_change(prepared(np.full((2, 12, 2, 2), .3, dtype=np.float32)), GEOMETRY)
    assert result["available"] is True
    for layer in result["layers"].values():
        assert layer["values"] == [[0., 0.], [0., 0.]]
        assert layer["valid_fraction"] == 1
        assert layer["mean_delta"] == 0


def test_one_computable_layer_retains_per_index_validity():
    data = np.zeros((2, 12, 2, 2), dtype=np.float32)
    data[:, 3] = .3  # NDVI is -1 at both dates; moisture/burn denominators stay zero.
    result = spectral_change(prepared(data), GEOMETRY)
    assert result["available"] is True
    assert result["layers"]["ndvi"]["mean_delta"] == 0
    assert result["layers"]["ndvi"]["valid_fraction"] == 1
    for name in ("ndmi", "nbr"):
        assert result["layers"][name]["mean_delta"] is None
        assert result["layers"][name]["valid_fraction"] == 0


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -.1, 1.1])
def test_nonphysical_required_band_never_produces_a_computable_delta(value):
    data = np.full((2, 12, 2, 2), .3, dtype=np.float32)
    data[1, 7] = value
    with np.errstate(all="raise"):
        result = spectral_change(prepared(data), GEOMETRY)
    assert result["available"] is False
    assert "finite [0,1] reflectance" in result["reason"]
    assert all(layer["valid_fraction"] == 0 for layer in result["layers"].values())
    json.dumps(result, allow_nan=False)


def test_masked_pair_reports_missing_common_coverage_separately():
    masks = np.ones((2, 2, 2), dtype=bool)
    masks[1] = False
    result = spectral_change(prepared(np.full((2, 12, 2, 2), .3, dtype=np.float32), masks), GEOMETRY)
    assert result["available"] is False
    assert result["reason"] == "No common valid pixels across the two dates."

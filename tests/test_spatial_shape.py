"""Classified patch shapes use measured, complete boundaries only."""
import math

import numpy as np
import pytest

from forestmind.monitoring.spatial import fragmentation_metrics, polygon_mask


GEOMETRY = {"type": "Polygon", "coordinates": [[[0., 0.], [.008, 0.],
             [.008, .008], [0., .008], [0., 0.]]]}


def measure(forest, valid=None, geometry=GEOMETRY):
    return fragmentation_metrics({"forest": forest.tolist(),
        "valid_mask": (np.ones_like(forest) if valid is None else valid).tolist(),
        "source": "Controlled classified grid fixture", "observed_at": "2024-01-01",
        "classification": "Explicit forest/nonforest"}, geometry)


def test_interior_square_has_analytic_grid_compactness_and_edge():
    forest = np.zeros((8, 8), dtype=bool)
    forest[3:5, 3:5] = True
    result = measure(forest)
    shape = result["shape"]
    patch = shape["patches"][0]
    assert shape["complete_patch_count"] == 1
    assert shape["censored_patch_count"] == 0
    assert patch["boundary_complete"]
    assert patch["compactness"] == pytest.approx(math.pi / 4, rel=1e-6)
    assert patch["area_km2"] == result["patch_area_km2"][0]
    assert patch["perimeter_m"] == pytest.approx(result["internal_edge_km"] * 1000)
    assert shape["mean_compactness"] == patch["compactness"]


def test_equal_area_elongated_patch_is_less_compact():
    square = np.zeros((8, 8), dtype=bool)
    square[3:5, 3:5] = True
    elongated = np.zeros_like(square)
    elongated[3, 2:6] = True
    compact = measure(square)["shape"]["patches"][0]
    long = measure(elongated)["shape"]["patches"][0]
    assert long["area_km2"] == pytest.approx(compact["area_km2"], rel=1e-6)
    assert long["compactness"] == pytest.approx(4 * math.pi / 25, rel=1e-6)
    assert long["compactness"] < compact["compactness"]


@pytest.mark.parametrize("censor", ["edge", "nodata", "polygon"])
def test_unknown_boundary_never_produces_complete_perimeter(censor):
    forest = np.zeros((8, 8), dtype=bool)
    valid = np.ones_like(forest)
    geometry = GEOMETRY
    if censor == "edge":
        forest[0:2, 3:5] = True
    elif censor == "nodata":
        forest[3:5, 3:5] = True
        valid[3, 2] = False
    else:
        geometry = {"type": "Polygon", "coordinates": [[[0., 0.], [.008, 0.], [0., .008], [0., 0.]]]}
        forest[3:5, 2:4] = True
    result = measure(forest, valid, geometry)
    shape = result["shape"]
    assert shape["censored_patch_count"] == 1
    assert shape["complete_patch_count"] == 0
    assert shape["mean_compactness"] is None
    assert shape["patches"][0]["perimeter_m"] is None
    assert shape["patches"][0]["compactness"] is None
    assert shape["patches"][0]["observed_edge_m"] > 0


def test_nodata_hole_censors_but_observed_nonforest_hole_counts():
    forest = np.zeros((8, 8), dtype=bool)
    forest[2:6, 2:6] = True
    forest[3:5, 3:5] = False
    observed = measure(forest)["shape"]["patches"][0]
    valid = np.ones_like(forest)
    valid[3:5, 3:5] = False
    unknown = measure(forest, valid)["shape"]["patches"][0]
    assert observed["boundary_complete"]
    assert observed["compactness"] == pytest.approx(math.pi / 12, rel=1e-6)
    assert not unknown["boundary_complete"]
    assert unknown["observed_edge_m"] < observed["perimeter_m"]


def test_diagonal_patches_stay_separate_and_empty_cover_is_finite():
    forest = np.zeros((8, 8), dtype=bool)
    forest[3, 3] = forest[4, 4] = True
    result = measure(forest)
    assert result["patch_count"] == 2
    assert result["shape"]["complete_patch_count"] == 2
    assert [p["patch_id"] for p in result["shape"]["patches"]] == [1, 2]
    assert result["shape"]["mean_compactness"] == pytest.approx(math.pi / 4, rel=1e-6)
    empty = measure(np.zeros_like(forest))["shape"]
    assert empty["patches"] == [] and empty["mean_compactness"] is None
    assert empty["complete_patch_count"] == empty["censored_patch_count"] == 0


def test_fully_forested_aoi_has_unknown_shape_instead_of_false_compactness():
    result = measure(np.ones((8, 8), dtype=bool))
    patch = result["shape"]["patches"][0]
    assert result["internal_edge_km"] == 0
    assert patch["observed_edge_m"] == 0
    assert patch["perimeter_m"] is None and patch["compactness"] is None


def test_nonrectangular_coverage_counts_only_sample_centres_inside_the_aoi():
    geometry = {"type": "Polygon", "coordinates": [[[0., 0.], [.008, 0.], [0., .008], [0., 0.]]]}
    forest = np.zeros((8, 8), dtype=bool)
    inside = polygon_mask(geometry, forest.shape)
    complete = measure(forest, geometry=geometry)
    assert complete["available"] and complete["patch_count"] == 0
    assert complete["forest_fraction"] == 0  # fully observed nonforest is a real zero
    assert complete["valid_fraction"] == 1
    assert complete["aoi_sample_count"] == complete["valid_sample_count"] == int(inside.sum())
    partial_mask = np.ones_like(forest)
    missing_rows, missing_cols = np.nonzero(inside)
    partial_mask[missing_rows[:3], missing_cols[:3]] = False
    partial = measure(forest, partial_mask, geometry)
    assert partial["valid_fraction"] == pytest.approx((int(inside.sum()) - 3) / inside.sum())
    # Values outside the AOI neither improve its coverage nor contribute forest.
    outside_forest = ~inside
    same = measure(outside_forest, partial_mask, geometry)
    assert same["valid_fraction"] == partial["valid_fraction"]
    assert same["forest_area_km2"] == 0


@pytest.mark.parametrize("valid_outside_only", [False, True])
def test_no_valid_samples_in_aoi_withholds_topology_instead_of_reporting_zero_forest(valid_outside_only):
    geometry = {"type": "Polygon", "coordinates": [[[0., 0.], [.008, 0.], [0., .008], [0., 0.]]]}
    forest = np.ones((8, 8), dtype=bool)
    inside = polygon_mask(geometry, forest.shape)
    valid = ~inside if valid_outside_only else np.zeros_like(forest)
    result = measure(forest, valid, geometry)
    assert not result["available"]
    assert result["valid_fraction"] == 0 and result["valid_sample_count"] == 0
    assert result["aoi_sample_count"] > 0
    assert "No valid classified samples" in result["reason"]
    assert result["source"] == "Controlled classified grid fixture"
    assert "patch_count" not in result and "forest_area_km2" not in result and "shape" not in result


def test_grid_without_aoi_sample_centres_requires_finer_classification():
    geometry = {"type": "Polygon", "coordinates": [[[0., 0.], [.008, 0.], [0., .008], [0., 0.]]]}
    result = measure(np.ones((1, 1), dtype=bool), geometry=geometry)
    assert not result["available"] and result["aoi_sample_count"] == 0
    assert "finer classified grid" in result["reason"]
    assert "patch_count" not in result

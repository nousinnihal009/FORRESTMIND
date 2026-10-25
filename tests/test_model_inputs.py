"""Registered inputs retain measured coverage and cannot change model semantics."""
from copy import deepcopy
import json
import numpy as np
import pytest

from forestmind.data.sequence import add_sequence_context, prepare_sequence
from forestmind.monitoring.model_inputs import modality_contract_reasons, select_model_sequence
from forestmind.monitoring.workflow import WorkflowService, digest
from forestmind.monitoring.analysis import analyze_region
from test_context_alignment import optical_cache, radar, climate, DATES, BBOX
from test_workflow import saved


def sequences(tmp_path):
    for index, day in enumerate(DATES):
        optical_cache(tmp_path, f"scene-{index}", day)
    base = prepare_sequence(tmp_path, through="2024-03-10", bbox=BBOX)
    return base, add_sequence_context(base, radar_context=radar(), climate_context=climate())


def contract(prepared):
    return {"required_modalities": ["optical", "sar"], "sar_units": "gamma0_db",
            "context_alignment": prepared.provenance["alignment"]["parameters"],
            "min_modality_fraction": {"sar": .5}}


def test_registered_coverage_units_and_alignment_are_enforced(tmp_path):
    _, prepared = sequences(tmp_path)
    accepted = contract(prepared)
    assert modality_contract_reasons(accepted, prepared) == []
    strict = deepcopy(accepted)
    strict["min_modality_fraction"]["sar"] = 1
    assert any("50.0%" in reason for reason in modality_contract_reasons(strict, prepared))
    changed = deepcopy(accepted)
    changed["sar_units"] = "sigma0_db"
    assert any("units" in reason for reason in modality_contract_reasons(changed, prepared))
    prepared.provenance["alignment"]["parameters"]["radar_max_age_days"] = 100
    assert any("hash changed" in reason for reason in modality_contract_reasons(accepted, prepared))


@pytest.mark.parametrize("required", [[], ["sar"], [None], [["optical"]], ["optical", "optical"]])
def test_malformed_or_radar_only_model_contracts_abstain(tmp_path, required):
    _, prepared = sequences(tmp_path)
    assert modality_contract_reasons({"required_modalities": required}, prepared)


def test_missing_masks_and_zero_observations_cannot_bypass_gate(tmp_path):
    _, prepared = sequences(tmp_path)
    accepted = contract(prepared)
    prepared.arrays["observation_mask"][:] = False
    assert modality_contract_reasons(accepted, prepared)
    prepared.arrays["observation_mask"][:] = True
    prepared.arrays.pop("sar_quality_mask")
    assert any("quality" in reason for reason in modality_contract_reasons(accepted, prepared))


def test_old_optical_model_gets_identical_numeric_inputs_without_context(tmp_path):
    base, prepared = sequences(tmp_path)
    selected = select_model_sequence(prepared, {"required_modalities": ["optical"]})
    for name in ("optical", "sar", "timestamps", "observation_mask", "optical_mask", "sar_mask", "quality_mask"):
        assert np.array_equal(selected.arrays[name], base.arrays[name])
    assert "climate_context" not in selected.arrays
    assert "sar_units" not in selected.arrays
    assert selected.provenance["model_input_modalities"] == ["optical"]
    assert prepared.arrays["sar_mask"].any()  # Original evidence remains unchanged.


def test_connected_workflow_aligns_context_and_preserves_immutable_snapshot(tmp_path, saved):
    store, region, observations = saved
    for row, day in zip(observations, DATES):
        row["date"] = day[:10]
    with store.connection() as db:
        db.execute("UPDATE regions SET observations=? WHERE id=?", (json.dumps(observations), region["id"]))
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    sequences(cache_root)
    service = WorkflowService(store)
    try:
        radar_context = radar()
        radar_context.update(geometry_sha256=digest(region["geometry"]), end="2024-02-11")
        service.save_context(region["id"], "radar", radar_context)
        result = service.run(region["id"], cache_dir=cache_root,
                             source_paths=list((cache_root / "sentinel2").glob("*.npz")), execute_model=False)
        assert result["sequence"]["available_modalities"] == ["optical", "sar"]
        assert result["sequence"]["alignment"]["radar"]["matched_steps"] == 1
        assert result["alignment_errors"] == {}
        original_id = result["manifest"]["id"]
        radar_context["grid_shape"] = [99, 99]
        service.save_context(region["id"], "radar", radar_context)
        changed = service.run(region["id"], execute_model=False)
        assert "radar" in changed["alignment_errors"]
        assert changed["manifest"]["id"] != original_id
        with store.connection() as db:
            original = db.execute("SELECT payload FROM workflow_snapshots WHERE id=?", (original_id,)).fetchone()[0]
        assert '"matched_steps":1' in original.replace(" ", "")
    finally:
        service.executor.shutdown(wait=True)

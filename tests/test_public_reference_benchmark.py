"""Controlled fixtures test the separate retrospective experiment, not accuracy."""
from copy import deepcopy

import numpy as np
import pytest

from forestmind.training import public_reference_benchmark as pb


def fixture_packet():
    samples = []
    for i, lon in enumerate(range(-172, 173, 10)):
        for j, lat in enumerate(range(-62, 63, 10)):
            index = i * 13 + j
            loss = 2006 + index % 15
            samples.append({"id": f"fixture-{index}", "lon": float(lon), "lat": float(lat),
                            "annual_state": {str(y): y < loss for y in range(2000, 2021)},
                            "annual_loss": {str(y): int(y == loss) for y in range(2000, 2021)},
                            "attributes": {"future_loss_year": loss}})
    return {"metadata": {"synthetic_test_fixture": True}, "samples": samples, "audit": {}}


def test_features_never_read_values_after_origin_or_source_summary_fields():
    sample = fixture_packet()["samples"][0]
    expected = pb.past_features(sample, 2005)
    changed = deepcopy(sample)
    for year in range(2006, 2021):
        changed["annual_loss"][str(year)] = None
        changed["annual_state"][str(year)] = None
    changed["attributes"] = {"future_loss_year": 2099, "stratum": "other"}
    assert pb.past_features(changed, 2005) == expected
    changed["annual_state"]["2004"] = None
    assert pb.past_features(changed, 2005) is None


def test_geography_is_label_blind_buffered_and_identical_per_sample():
    packet = fixture_packet()
    rows, audit = pb.build_examples(packet)
    assert audit["nearest_between_split_centres_km"] > 10.04
    assert all(n > 5 for n in audit["split_counts"].values())
    splits = {}
    for row in rows:
        splits.setdefault(row["sample_id"], set()).add(row["split"])
        assert row["target_year"] == row["cutoff_year"] + 1
    assert all(len(v) == 1 for v in splits.values())
    assert pb.spatial_assignment(10., 20.)[2] is False
    assert pb.spatial_assignment(12., 22.)[2] is True
    assert pb.spatial_assignment(12., 89.99)[2] is False
    packet["samples"].reverse()
    assert pb.build_examples(packet) == (rows, audit)


def test_unknown_outcomes_are_excluded_and_do_not_become_negative():
    packet = fixture_packet()
    before, _ = pb.build_examples(packet)
    sample_id = before[0]["sample_id"]
    sample = next(s for s in packet["samples"] if s["id"] == sample_id)
    sample["annual_loss"][str(before[0]["target_year"])] = None
    after, audit = pb.build_examples(packet)
    assert before[0]["id"] not in {r["id"] for r in after}
    assert audit["excluded"]["origins_with_unknown_outcome"] >= 1


@pytest.mark.parametrize("mutation", ["id", "coordinates", "invalid_coordinate"])
def test_bad_or_duplicate_spatial_identity_is_rejected(mutation):
    packet = fixture_packet()
    if mutation == "id":
        packet["samples"][1]["id"] = packet["samples"][0]["id"]
    elif mutation == "coordinates":
        packet["samples"][1].update(lon=packet["samples"][0]["lon"], lat=packet["samples"][0]["lat"])
    else:
        packet["samples"][0]["lat"] = float("nan")
    with pytest.raises(ValueError):
        pb.build_examples(packet)


def test_metrics_known_values_and_missing_event_day():
    result = pb.binary_metrics([0, 1, 0, 1], [.1, .8, .6, .3])
    assert result["brier"] == pytest.approx(.225)
    assert result["auroc"] == .75
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["expected_calibration_error"] == pytest.approx(.4)
    assert result["lead_time_days"] is None
    result = pb.binary_metrics([0] * 6, [0, .2, .4, .6, .8, 1])
    assert sum(b["count"] for b in result["reliability"]) == 6
    assert result["auroc"] is None and result["average_precision"] is None


def test_block_resampling_keeps_dependent_origins_and_negative_results():
    rows = [{"block": "one", "target": 0} for _ in range(100)]
    assert not pb.block_improvement(rows, np.zeros(100), np.ones(100))["available"]
    rows = [{"block": str(i), "target": 0} for i in range(6) for _ in range(i + 1)]
    result = pb.block_improvement(rows, np.zeros(len(rows)), np.ones(len(rows)))
    assert result["lower"] == result["upper"] == -1.


def test_protocol_integrity_and_runtime_are_required_before_fit():
    packet = fixture_packet()
    protocol = pb.freeze_public_protocol(packet)
    assert "positive_counts" not in protocol["audit"]
    changed = deepcopy(packet)
    changed["samples"][0]["annual_loss"]["2020"] = 1
    with pytest.raises(ValueError, match="differs from frozen"):
        pb.run_public_benchmark(changed, protocol)
    changed_protocol = deepcopy(protocol)
    changed_protocol["runtime"]["numpy"] = "different"
    changed_protocol["protocol_sha256"] = pb._hash({k: v for k, v in changed_protocol.items() if k != "protocol_sha256"})
    with pytest.raises(ValueError, match="runtime differs"):
        pb.run_public_benchmark(packet, changed_protocol)


def test_real_estimator_pipeline_preserves_test_ids_and_no_operational_claim():
    packet = fixture_packet()
    protocol = pb.freeze_public_protocol(packet)
    result = pb.run_public_benchmark(packet, protocol)
    assert len(result["models"]) == 10
    ids = [r["id"] for r in result["test_predictions"]]
    assert len(ids) == len(set(ids)) == protocol["audit"]["split_counts"]["test"]
    assert all(set(r["scores"]) == set(result["models"]) for r in result["test_predictions"])
    assert result["deployable_checkpoint"] is None
    assert not result["scientifically_validated_operational_model"]
    assert not result["validation_used_for_model_selection"]
    assert result["target"]["historical_availability_verified"] is False


def test_outputs_refuse_overwrite(tmp_path):
    path = tmp_path / "protocol.json"
    pb.write_new_record({"value": 1}, path)
    with pytest.raises(FileExistsError):
        pb.write_new_record({"value": 2}, path)
    assert '"value": 1' in path.read_text()

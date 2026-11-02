# Reference data for ForestMind's first predictive study

## What would help from you

You do not need to provide a trained model. Useful contributions are dated forest-department surveys, reviewed loss polygons, field observations, or access to a qualified person who can check annotations. Retain the boundary/coordinates, observation dates, measurement method and uncertainty, original files, provider/version, publication date, permission to use the data, reviewer and review rationale. Distinguish natural forest from plantations and other tree-covered land uses.

Unlabeled imagery supports observation but does not supply confirmed health, driver or future-outcome labels. The user confirmed that no independently verified reference dataset is available. No real model training or ecological validation has been performed in this milestone.

## Primary target: annual gross tree-cover-loss fraction

The versioned protocol `annual-gross-tree-cover-loss-v1` is defined in `forestmind/data/reference_review.py` and included verbatim in every intake.

| Choice | Definition |
| --- | --- |
| Scope | Independently reviewed stand-replacement tree-cover disturbance. It may be temporary or permanent and does not identify cause. |
| Spatial unit | Fixed 100 ha (1 km²) WGS84 rectangular study patch; computed spherical area within 5% tolerance. Existing monitoring boundaries remain unchanged. |
| Baseline | Independently documented tree-covered area at cutoff: trees taller than 5 m, canopy cover at least 30%, at least 10 ha within the patch. Do not infer it from NDVI. |
| Horizon | December 31 cutoff through the following December 31. Freeze 365- and 366-day years separately. No invented exact event day. |
| Continuous target | Reviewed loss area divided by baseline tree-covered area, on a 0–1 scale. This is not a condition score and must not enter the existing health trajectory head. |
| Significant loss | At least 10% of baseline tree-covered area and at least 1 ha. These are study choices, not universal ecological thresholds or demonstrated sensor detection accuracy. |
| Positive | The entire reviewed loss-area uncertainty interval meets that criterion. |
| Negative | The entire interval lies below the criterion. This means below the criterion, not undisturbed or healthy. |
| Uncertain | Missing/pending/rejected assessment, incomplete coverage, insufficient baseline forest, excessive uncertainty, or an interval straddling the criterion. Excluded from supervised targets. |
| Quality | Complete reference coverage and supplied uncertainty no larger than 5% of baseline tree-covered area. An assessment bound is not automatically a statistical confidence interval. |

Review these choices before freezing a real study. A different definition needs a new versioned protocol and tests; changing an existing target is refused. Selecting only sufficiently certain labels affects the evaluation population: retain and report excluded examples.

## Public products provide candidate evidence

[Hansen/UMD Global Forest Change 2025 v1.13](https://storage.googleapis.com/earthenginepartners-hansen/GFC-2025-v1.13/download.html) provides annual loss layers under CC BY 4.0. Its provider warns about sensor/algorithm changes and treating raw pixel counts as definitive area estimates. Preserve version, attribution and uncertainty, with independent review or an appropriate sampling design. A public raster is not automatically a verified field label.

[WRI's explanation](https://gfr.wri.org/annual-tree-cover-loss-data-explained) distinguishes tree-cover loss from permanent deforestation and explains the year-2000 baseline and first-loss recording. This does not automatically establish current baseline cover, repeated disturbances or regeneration. The intake accepts supplied local evidence; it does not download these products or certify accuracy.

## Create and check a packet

```powershell
python scripts/review_references.py init --output data/my-reference-study/intake.json
python scripts/review_references.py check --manifest data/my-reference-study/intake.json
python scripts/review_references.py prepare --manifest data/my-reference-study/intake.json --output artifacts/my-reference-review-v1
```

`init` creates an empty template without labels. `check` reads only and exits with code 2 while the study is not ready. `prepare` refuses existing output directories. It saves decisions, exclusions, the source-manifest snapshot, source location and integrity hashes. It writes `references.json` only after the eligible labels pass sample-count, geographic and chronological audits. Readiness means software contracts passed, not independently certified label quality or test custody.

Each example needs the existing baseline-manifest fields: ID, region ID, rectangle, split, cutoff, target date, feature availability, feature values/nulls, quality group, forest type and dated baseline predictions. Add the following assessment. All placeholders below must be replaced with real values; it is intentionally not a valid label.

```json
{
  "reference": {
    "bbox": [null, null, null, null],
    "period_start": "YYYY-01-01",
    "period_end": "YYYY-12-31",
    "baseline_forest_area_ha": null,
    "loss_area_ha": null,
    "uncertainty_area_ha": null,
    "coverage_fraction": null,
    "source": {
      "independent": true,
      "reference": "Actual survey or reviewed product identifier",
      "version": "Actual version",
      "license": "Actual permission to use the evidence",
      "method": "Actual measurement method",
      "uncertainty": "How the area uncertainty was established",
      "independence_basis": "Why the reference is independent of the prediction heuristic",
      "available_at": "YYYY-MM-DD",
      "file": "evidence/original-file",
      "sha256": "Actual file SHA256"
    }
  },
  "review": {"status": "pending"}
}
```

Missing assessments may use `reference: null`. Accepted/rejected reviews need `reviewer`, `reviewed_at` and `basis`; review cannot predate the evidence. Paths, including resolved symlinks, must stay inside the intake directory. Hashes are checked with limits of 512 MiB per file and 2 GiB per intake. Keep original evidence alongside exported reports; the report records its location. Hashes detect changes, not truth.

An optional `baseline_history` contains 3–50 consecutive prior annual loss fractions, each with `period_end`, `available_at`, `loss_fraction` and the same hash-pinned `source` metadata. The converter computes persistence, annual seasonal-naive (equal to persistence here) and a least-squares historical trend clipped to 0–1. Outcomes and publications must be available by cutoff; latest history may be at most two years old. No absent year is filled. Conflicting supplied baseline values are rejected. Alternatively supply explicitly dated baseline predictions; their derivation remains the researcher's responsibility.

After enough references pass:

```powershell
python scripts/benchmark_references.py freeze --input artifacts/my-reference-review-v1/references.json --protocol artifacts/my-study-protocol.json
python scripts/benchmark_references.py run --input artifacts/my-reference-review-v1/references.json --protocol artifacts/my-study-protocol.json --output artifacts/my-study-comparison.json
```

Use separated region identities, buffered footprints and later periods for train, optional validation, calibration and test. Reference publication lag participates in chronological separation. The default 10 km buffer is a study assumption, not an established independence distance. The 10/5/5 sample minima, plus 5 for optional validation, are software minima rather than adequate scientific sample sizes. Existing frozen protocols pin their implementation hash: restore the original code/runtime for an old study; do not refreeze inspected test data simply to bypass a mismatch.

## Remaining evidence

The current pilot has too little history and geographic breadth. We need reviewed positive and negative references across separated locations and years. Foundation-model comparisons, spatial probabilities, condition/driver labels, ecological recovery and causal interventions require additional evidence and implementation. This annual continuous target does not fill those gaps.

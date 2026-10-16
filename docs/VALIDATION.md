# Build verification

ForestMind v0.5 verified locally on Windows on 2026-10-16, with the final monthly-history artifact and persistence audit on October 17. The first real single-scene pilot was verified during v0.4 on October 14.

## Completed checks

- Full test suite: **472 passed** in 64.61 seconds, including raster reprojection, strict inference and API contracts, acquisition-to-sequence conversion, evaluation provenance, explicit satellite providers, resumable monthly acquisition, report integrity, coverage/readiness, download failures, persistence, monitoring and saved plots.
- Frontend syntax: `app.js`, `model-runs.js` and `coverage.js` passed `node --check`.
- Browser: checked region selection, alerts and observations, JSON import validation, scenario calculation and invalidation after input changes, historical backtesting, and responsive layouts. No browser console errors were observed in the final preview.
- CPU training and evaluation: the small synthetic configuration completed and produced `checkpoints/quick_test/best_model.pt`.
- Checkpoint demo: completed successfully with the configured optical input shape of 4 time steps by 12 bands by 32 by 32 pixels.
- Inference: the actual quick-test checkpoint produced `artifacts/demo-inference-report.json` from the explicitly synthetic `artifacts/demo-inference-input.npz`. The report preserves synthetic input and training provenance and all five configured forecast horizons.
- Model-run browser flow: imported that report into the Tapajós demo region, verified malformed JSON feedback, reloaded the page and confirmed persistence. Narrow-screen layout had no page overflow, and the browser console had no errors. Screenshot: `artifacts/forestmind-model-runs.jpg`.
- Real-observation browser flow: imported `artifacts/real-pilot-region.json` as **Tapajos real satellite pilot** (`import-9e78ebc0e30c`). Verified the acquisition date, spectral indices, 100% valid samples and Earth Search source. Forecasts and resilience diagnostics correctly remain unavailable for the one-observation record. No browser console errors. Screenshot: `artifacts/forestmind-real-pilot.jpg`.
- Coverage browser flow: selected a twelve-month window for the saved single observation and verified one usable month, eleven missing months and no usable run at the requested period end. Checked JSON export, reversed-date feedback and clearing stale results when dates change. At a 390-pixel viewport there was no horizontal page overflow. No browser console errors.
- Monthly-history browser flow: imported eight actual acquisitions as **Tapajos monthly satellite pilot** (`import-2973ec681f83`). After a server restart, verified persistence and the July 2024–June 2025 coverage window: eight usable months, four missing months, longest usable run of seven and a run of one at the period end. Forecast and resilience results remain unavailable. The final preview had no browser console errors. Screenshot: `artifacts/forestmind-history-coverage.jpg`; API response: `artifacts/history-coverage.json`.
- Evaluation: `artifacts/quick-test-evaluation.json` contains raw predictions and reference targets for two synthetic test examples, configured horizon errors and reproducibility hashes. These sample metrics are not real-world validation.
- Packaging: built `artifacts/dist/forestmind-0.5.0-py3-none-any.whl` and verified that it contains the history acquisition and coverage modules, coverage JavaScript, inference runner, sequence preparation and model-results JavaScript. SHA256: `63d599837dbd60df468fc8afca679eae40bcc40b3a00ef1d2509c8c9d0660fc6`.

The optional raster dependencies used for the full suite were installed under the ignored local directory `artifacts/geo-validation`. They are not bundled in the package. The test invocation was:

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'artifacts/geo-validation'
python -m pytest tests -q --tb=short
```

## Run locally

From the repository root, after installing the dependencies described in the README:

```powershell
python scripts/serve.py --port 8000
```

Open <http://127.0.0.1:8000>. The preview server used for verification is temporary; start this command for a new session.

## Live-data verification

The public Planetary Computer catalog returned the actual scene `S2B_MSIL2A_20240707T140709_R110_T21MYS_20240707T174548`, acquired on 2024-07-07, for the small demonstration AOI `[-55.10,-3.50,-55.08,-3.48]`.

Planetary Computer raster ingestion did not complete. Bounded reads returned incomplete HTTP 206 byte ranges. A small 16 KiB range request succeeded, but serial GDAL requests still produced truncated raster tiles. The speculative serial-request change was reverted. Failed reads return a nonzero status and do not produce a usable cache. No real observations or model assessments were fabricated from these failures.

`artifacts/real-pilot-range-diagnostic.json` records the Planetary Computer range-read diagnostic. Download logs redact signed access queries, including truncated source strings emitted by the raster decoder.

The provider check exposed a separate reflectance-processing issue: newer processing baselines require a nonzero offset when asset metadata omit scaling. This is corrected in cache version `forestmind-grid-v3`; older caches are rejected by both observation and sequence preparation. Tests cover baseline-based scaling, including reprocessed historical scenes.

### Successful Earth Search pilot

An explicitly selected alternate provider, Element 84 Earth Search, completed the ordinary download CLI in 99.0 seconds. It returned **`S2A_T21MYS_20240729T140134_L2A`**, acquired **2024-07-29T14:03:30.961Z**, from `sentinel-2-c1-l2a` for the same AOI. All twelve optical bands and the SCL mask were read and aligned to a 64 × 64 grid. All **4,096 sampled pixels** passed the masks. The scene-wide cloud metadata was 1.466749%; that is distinct from validity within this small AOI. Each band's explicit metadata specified `DN × 0.0001 − 0.1`, applied once.

Prepared observations were NDVI **0.852677**, NDMI **0.291797** and NBR **0.631346**. The import JSON and its provenance are saved as `artifacts/real-pilot-region.json` and `artifacts/real-pilot-region.provenance.json`. The dashboard displays these real measurements alongside four clearly identified synthetic demo regions.

`artifacts/real-pilot-sequence.npz` and its provenance sidecar contain one actual optical acquisition, its masks, timestamps and source hashes. The cutoff was July 31; the last actual observation remains July 29. SAR and labels are absent, and no dates were padded. This sequence was not passed through the synthetic demonstration checkpoint as a real ecological forecast.

This verifies a real provider → aligned cache → observation preparation → dashboard path and a cache → inference-input path. It does not verify a long monitoring history, field labels or predictive skill. The exact pilot commands are in the README; the full download and preparation evidence is in `artifacts/real-pilot-verification.json`.

### Resumable monthly-history pilot

The monthly planner searched **2024-07-01 through 2025-06-30** for the same small AOI and explicit Earth Search collection. The frozen plan uses a 64 × 64 grid, at most three candidates per month from a bounded search of thirty, a minimum valid-pixel fraction of 0.6, and per-invocation limits of six attempts and two new downloads. Candidate ranking and the scene-cloud filter are recorded in the plan; this is a selected sample, not an exhaustive history of every available acquisition.

Four execution invocations, with `--resume` after the first, each made exactly **two attempts and two downloads**. They produced eight accepted observations. Earlier accepted caches were retained and verified during resume. February–May 2025 had no eligible candidates in the frozen search results, so the final report correctly remains **incomplete** with `candidates_exhausted`; absent months were not filled. This does not establish that no satellite observations exist for those months under other search criteria.

| Actual acquisition date | Valid pixel samples |
| --- | ---: |
| 2024-07-17 | 100% |
| 2024-08-26 | 100% |
| 2024-09-02 | 100% |
| 2024-10-12 | 99.707% |
| 2024-11-29 | 100% |
| 2024-12-01 | 64.087% |
| 2025-01-23 | 76.953% |
| 2025-06-16 | 100% |

The final audit validated the plan and report hashes, all eight accepted cache files, their source identities, processing grids, bands and quality masks. Recomputing observations from the accepted files reproduced `artifacts/history-region.json` exactly and matched the deterministic provenance fields. The earlier July 29 single-scene pilot is excluded from this separate monthly export. A mask passing the software checks is not field verification of cloud-free or ecologically representative measurements.

Saved evidence:

- `artifacts/history-plan.json`: frozen search results, settings and plan hash.
- `artifacts/history-progress.json`: schema-2 report, four runs, accepted source hashes and missing-month outcomes.
- `artifacts/history-region.json` and `artifacts/history-region.provenance.json`: eight actual observations and their source audit trail.
- `artifacts/history-coverage.json`: persisted region's API response for the complete requested year.
- `artifacts/forestmind-history-coverage.jpg`: final dashboard coverage preview.

The README documents the reproducible `build_history.py plan` and `execute --resume` workflow. This pilot establishes bounded downloads, resume integrity and an honest gap report for real data. It has only seven consecutive usable months and does not meet the current software minimums of twelve for a baseline forecast or thirty-six for descriptive resilience diagnostics.

## Research limits

These checks establish software behavior, not ecological predictive validity. The four seeded monitoring regions and quick training run use synthetic data. The two imported real pilots contain one observation and eight monthly observations respectively; the monthly pilot has four gaps in its requested year and remains insufficient for forecasts or resilience diagnostics. Native pretrained checkpoints and operational deployment were not exercised. The new offline inference command rejects native backbone constructors that would download weights; only local custom/proxy checkpoints are currently supported there. Monitoring scores and scenario outputs remain uncalibrated research indicators. Validation on independent real observations and events, geographic and temporal holdouts, and uncertainty calibration remain necessary before scientific or operational claims.

See `METHODOLOGY.md`, `RESEARCH_PROTOCOL.md` and `COMPLETION_PLAN.md` for the assumptions and remaining milestones.

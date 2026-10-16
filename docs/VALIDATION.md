# Build verification

ForestMind v0.5 verified locally on Windows on 2026-10-16. The first real single-scene pilot was verified during v0.4 on October 14.

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

## Research limits

These checks establish software behavior, not ecological predictive validity. The four seeded monitoring regions and quick training run use synthetic data; the added real pilot has only one observation. Native pretrained checkpoints and operational deployment were not exercised. The new offline inference command rejects native backbone constructors that would download weights; only local custom/proxy checkpoints are currently supported there. Monitoring scores and scenario outputs remain uncalibrated research indicators. Validation on independent real observations and events, geographic and temporal holdouts, and uncertainty calibration remain necessary before scientific or operational claims.

See `METHODOLOGY.md`, `RESEARCH_PROTOCOL.md` and `COMPLETION_PLAN.md` for the assumptions and remaining milestones.

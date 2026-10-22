# Build verification

## v0.6 connected application — October 22, 2026

This release continues the existing FastAPI/SQLite/buildless dashboard. It connects worldwide search, small confirmed AOIs, actual acquisition, durable jobs, saved observations, optional real context, reproducible analyses, refresh and local schedules. The full original scientific proposal is **not complete**; the [requirements matrix](REQUIREMENTS_MATRIX.md) maps all 59 original-proposal and 40 expanded acceptance items, including incomplete implementations and missing evidence.

### Executed software checks

- Final integrated regression: **671 passed in51.69 seconds**, with geospatial dependencies enabled. Command: `python -m pytest tests -q --tb=short --basetemp=artifacts/pytest-release-20261022`. Log: `artifacts/regression-0.6.0-release.txt`. All tests passed; none were skipped.
- Earlier failures remain recorded: the initial v0.6 run had611 passes and three WinError5 atomic-save failures (`artifacts/regression-0.6.0.txt`). The bounded Windows retry repair then passed a623-test full run. After the additional diagnostics, an overlapping run had670 passes and one duplicate-ID error in a newly added test fixture (`artifacts/regression-0.6.0-20261022.txt`); that fixture was corrected before the final671-test run. Neither earlier failure is represented as a pass.
- Focused acquisition/history/WorldCover/radar checks:144 passed before that repair. Workflow/reference/API/model-import checks:70 passed; final additional capacity/freshness checks:13 passed; reference-experiment checks:15 passed. Native inference/adapter checks:83 passed. Notifications/search/local-security/full-backup checks:18 passed. Frontend contracts:14 passed. These runs overlap and must not be added together as a total.
- All seven dashboard JavaScript files passed `node --check`.
- Final focused additions:14 notification tests,20 reference-experiment tests,38 climate/workflow/frontend tests,9 spectral-availability tests and46 spatial/workflow/landcover tests passed. These overlap the full suite. The Windows repair also passed an actual non-delete-sharing file-handle test.
- Packaging: rebuilt `artifacts/dist/forestmind-0.6.0-py3-none-any.whl`,287,363 bytes, SHA256 `c8b95a0e2f6595228a18ca698ee4e99fc31e69b5d737630877ece0c54e788d13`. All82 packaged source/assets match current files byte-for-byte (`artifacts/package-verification-0.6.0.json`). Installed the wheel into a separate local directory and loaded its actual package; its empty-workspace API, HTML and updated JavaScript returned200 (`artifacts/wheel-smoke-0.6.0.json`).
- Runtime inventory: `artifacts/validated-runtime-0.6.0.json`, Windows/Python3.13.7, CPU PyTorch, with optional geospatial dependencies in `artifacts/geo-validation`. This is a record of tested versions, not a universal dependency lockfile or security audit.

### Actual browser workflow and provider results

The in-app browser performed the application actions below. No synthetic observations were substituted, no public deployment occurred, and no external notification was sent.

| Check | Observed result |
| --- | --- |
| Worldwide search | Explicit `Tapajos Brazil` query returned eight disambiguated Photon/OSM results with location and attribution. A place result was selected, then a small independent study rectangle was confirmed. |
| Confirm and collect | `Tapajos connected real pilot`, bbox `[-55.10,-3.50,-55.08,-3.48]`, July–August2024,64×64 grid, two-download cap. Job `acq-d74c8ccee2f14153` accepted July17 and August26,100% valid samples, and saved area `area-858ffd6a4f28`. |
| Refresh/cancel/resume | UI refresh job `acq-aead355d7b794983` was cancelled during planning, confirmed cancelled with saved observations unchanged, then explicitly resumed. It accepted September2 and October12 using exactly two downloads and retained the original July/August records. The broader requested window remains partial with explicit gaps. |
| Local schedule | Enabled a15-day interval in the UI, observed a next-run date, then disabled it after the check. Restart retained `enabled:false`, `interval_days:15`; no unattended test schedule remains enabled for this area. |
| Additional geography | Existing user area `Tamil Nadu, India` was preserved: November10,2025,95.9229% valid samples. New `Black Forest real pipeline check`, bbox `[8.20,48.55,8.22,48.57]`, saved August11,2024 at100% validity; July had no eligible candidate in the bounded plan. These are pipeline checks, not ecological geographic-transfer validation. |
| No data | A small Pacific Ocean test rectangle returned no eligible observations. The job is failed with month-level reasons and zero downloads; no saved region or fabricated measurement was created. |
| Climate context | UI ERA5 collection saved July17–August26,2024 centroid reanalysis, actual units/counts and partial-month date ranges. Standalone three-day smoke also succeeded (`artifacts/climate-context-pilot.json`). |
| Land-cover context | Actual ESA WorldCover2021 v200 crop,64×64,4096 valid samples, class10 throughout this small Tapajós area. UI computed one patch and zero internal edge; this is historical classified woody cover, not independent current forest verification. `artifacts/worldcover-real-smoke.json`. |
| Radar context | UI collected S1A/orbit39/descending RTC acquisitions August4 and16,2024,100% common valid samples; mean VV change≈+0.211dB and VH≈+0.451dB. A separate July11→23 smoke is retained in `artifacts/radar-real-smoke.json`. Dates/polarizations/units/orbit and masks are exported; the pairs do not establish disturbance cause. |
| Spatial/keyboard | Real NDVI/NDMI/NBR difference rasters render. ArrowRight changes optical and radar pixel readouts. Final optical example reads pixel2,1≈+0.00397 NDVI with area mean≈−0.064. It is measured spectral change, not loss or future-risk probability. |
| Mobile |390×844 viewport: document client/scroll widths both375px; no page overflow in real-region list or connected analysis. Spatial grid, labels and keyboard focus remained usable. Wide tables have internal scrolling. Screenshot `artifacts/forestmind-connected-mobile.jpg`. |
| Desktop |1365×900 breakpoint checked, selected-area identity/cutoff and connected controls visible. Screenshot `artifacts/forestmind-connected-desktop.jpg`. Final browser error log was empty after reconnecting to the running service. Temporary viewport overrides were reset. |
| Browser export | **Export reproducible record** downloaded the actual selected-area JSON. Workspace copy `artifacts/browser-connected-export.json`:2,327,832bytes, SHA256 `037b0a7e42f2f3baa5eae95cc809aa020ca06e49a68583995e3418381a7aaa00`; geometry/cutoff2024-10-12/four observations/context hashes/version0.6.0 retained, model null. |
| Persistence/immutability | After restart, all nine areas remained (four synthetic/five real records), including all four refreshed Tapajós observations. Export of the older exact snapshot ID compared equal to its saved pre-restart payload. Rerunning after code changes produced a fresh non-stale snapshot. `artifacts/restart-verification-0.6.0.json`. |
| Operations | UI showed local service and database integrity `ok`, available disk space, and created verified DB snapshot `forestmind-20261020T112518-8a22d2.db`. Notifications displayed disabled/unconfigured with zero outbox entries. |
| October22 connected rerun | After restarting with the final code, clicked **Run selected-area analysis** for the real Tapajós area. A new non-stale manifest `f52622afdec675a60940be1889023894d1d32fb233afb2ed26f7a3f2edbec98b` retained geometry, cutoff and four observations. Existing raw climate values/hash were unchanged; no additional provider collection was requested. |
| Paired climate diagnostic | Browser showed July17–31 precipitation minus reference ET0 **−73.34mm**,15/15 paired days, and August1–26 **−111.10mm**,26/26 paired days. Both are partial calendar months, explicitly distinct from validated drought or forest water use. `artifacts/forestmind-climate-20261022.jpg`. |
| Patch shape | Actual WorldCover crop has one patch reaching the AOI boundary. Browser correctly displayed0 complete/1 clipped-or-unknown boundaries, with null perimeter/compactness. Expanded table retained the4.9366km² classified area and explanation. Analytic complete-shape behavior is tested with controlled fixtures; no real complete-patch ecological validation is claimed. |
| Final responsive check | New climate/shape sections checked at desktop and390×844 mobile. Document client/scroll widths were1350/1350px and375/375px respectively; tables scroll internally. No browser console errors were recorded. Temporary viewport override was reset. `artifacts/forestmind-shape-mobile-20261022.jpg` and `artifacts/forestmind-release-20261022.jpg`. |
| Final browser export | Downloaded `artifacts/browser-connected-export-20261022.json`,2,335,851 bytes, SHA256 `f6771484df2a11691fe67756e8b0732f2988546307b26741d22ed1fe0af54148`. Geometry and observations equal the earlier export; the older exact manifest still returns its original payload. Detailed checks: `artifacts/final-browser-verification-20261022.json`. |

`artifacts/connected-final-jobs.json` retains the five acquisition jobs and their actual settings/outcomes. The original single-scene and eight-month imports remain unchanged. The real histories are still too short or discontinuous for the current forecast/resilience minimums, and the application withholds those outputs. Trained model and five-year ecological forecast validation were not demonstrated on these records.

### Recovery and scientific gates

Full workspace backups were created and hash/integrity verified at `data/backups/workspace-20261019` (11files) and `data/backups/workspace-20261020` (22files), including SQLite and both artifact trees. Automated restore tests recover the original job/source references and reject tampering, path traversal and overwrites. Database-only UI snapshots are deliberately distinguished from full source backups. Schedule concurrency fixes preserve user edits during enqueueing and prevent duplicate due ticks. The [operator guide](OPERATOR_GUIDE.md) documents original-path restore, quotas and rollback.

**G1 achieved for the documented bounded local monitoring workflow:** real browser search/acquisition, progress/cancel/resume, saved observations, connected context and diagnostics, refresh/schedule, failure/persistence/export, responsive checks, regression and installed-package smoke have passed. This acceptance does not cover every original research capability or guarantee worldwide data availability. **G2:** public/hosted deployment is not ready: no authentication, area/job ownership isolation, hosted supervision/load testing or real webhook-delivery acceptance. Local access is enforced. **G3:** not achieved: independent reference labels, real-trained accepted models, protected experiments and ecological predictive/causal evidence are missing. The five-baseline reference-experiment harness is tested with fixtures; it has not run an independent real-data benchmark. Native pretrained runtime/checkpoints remain unavailable and offline loading is blocked, as detailed in [NATIVE_MODEL_AUDIT.md](NATIVE_MODEL_AUDIT.md). Other incomplete original capabilities remain explicit in the matrix and completion plan.

## Earlier v0.5 evidence (retained historical record)

ForestMind v0.5 was verified locally on Windows on2026-10-16, with its monthly-history artifact and persistence audit on October17. The first real single-scene pilot was verified during v0.4 on October14. The counts and limitations below describe that earlier build.

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

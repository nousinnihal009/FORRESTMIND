# Continuation record — 2026-10-18

The user requests the full original proposal plus the worldwide search → real acquisition → monitoring workflow. Do not interpret the local application milestone as scientific completion. Original proposal: attachment `2356b8c3-679f-4302-9adf-414b4a571fcc/Pasted text.txt`; current expanded request: attachment `044b2d52-e600-4756-bd72-97e7c1ad4605/Pasted text.txt`.

## Work in progress

- Parent: application integration, Photon place search, local HTTP boundary, migrations/backups, alert lifecycle, regression/browser/live-data verification and operator documentation.
- Acquisition agent: persistent single-worker jobs, frozen history integration, cancellation/recovery/resume, budgets, saved area merge and schedules.
- Web agent: existing visual language plus worldwide map/search/area selection, job controls, refresh/schedules, connected results and alert handling.
- Analysis agent: immutable evidence manifests, automatic eligible local checkpoint execution, spatial change/recovery/fragmentation computations, requirements matrix.

The code at the beginning was v0.5 with 472 recorded tests. The first integrated regression run in this continuation passed **528 tests in 78.22s** before subsequent WorldCover/scenario/backup/notification additions. Focused acquisition/history/API tests passed **129 in 30.24s**. Search/security/alerts/backup/API checks initially passed **36 in 12.83s**. A subsequent focused run passed 63 tests and found one Windows backup file-handle leak; fixed using explicit connection closure, then both complete workspace restore tests passed. Final full-suite results are still pending after the additional changes.

No paid services, external notification delivery or deployment are authorized or performed. Notifications are now opt-in code with mocked tests; keep delivery off in this workspace. Existing real pilot and synthetic examples remain distinct.

## Actual browser and provider evidence so far

- Existing default database and user-created Tamil Nadu area were preserved. Tamil Nadu has one real November 2025 observation, explicit gaps and a partial two-download job.
- In the actual browser, explicit Photon query `Tapajos Brazil` returned disambiguated Brazilian places with OSM/Photon attribution. Selected a place, then confirmed the established small study rectangle `[-55.10,-3.50,-55.08,-3.48]` with an explicit July–August 2024 window and two-download cap.
- `Tapajos connected real pilot` was acquired through the UI and saved with **two real observations**, July 17 and August 26, 2024. Browser coverage showed 2/2 usable months and correctly withheld short-history forecasts/resilience. Job records saved in `artifacts/connected-jobs-before-restart.json`. Full connected-result/restart/refresh/schedule browser verification is still pending.
- Real ERA5 climate smoke succeeded for July 1–3, 2024; artifact `artifacts/climate-context-pilot.json`.
- Real ESA WorldCover 2021 64×64 crop succeeded for Tapajós, 4096 valid samples, all mapped tree-cover class10; `artifacts/worldcover-real-smoke.json`. This is an annual land-cover classification, not verified current ecological forest or land-loss evidence.
- Server started as process22896 on loopback port8000. Inspect current process command line and active jobs before restarting; do not assume this PID remains current. The browser tab has been preserved across usage interruptions.

## Environment note

Windows sandbox execution can hang while FastAPI TestClient constructs an asyncio socket pair. The focused and full test commands worked outside the sandbox after approval. This was diagnosed in faulthandler as socket.accept, before app code. Raster/STAC extras are in ignored `artifacts/geo-validation`; use its path as PYTHONPATH for complete raster tests and the local server. Avoid leaving stalled duplicate test/server processes.

## Provider choices and bounds

Photon explicit search, server cache, single outstanding request and at most one request per 1.1 seconds. Photon/OSM attribution; named places are not forest boundaries. Geocoder can be changed through operator configuration. Public demo service has no SLA.

Acquisition uses explicit Earth Search Sentinel-2, strict small WGS84 rectangles and recorded quality settings. Failed requests and missing observations remain visible. Network destinations restricted in web worker context. Local HTTP service rejects remote peers/unknown Host and cross-origin writes; public hosting remains disabled.

## Handoff rule

Read current files and this record before continuing; do not repeat completed work. Advance three separate gates: working local app, deployment readiness, scientific validity. Record exact unresolved code work as well as external reference-label/model/compute dependencies. Tests and synthetic demonstrations never establish ecological predictive validity.

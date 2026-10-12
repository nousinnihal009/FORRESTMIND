# Build verification

Verified locally on Windows on 2026-10-13.

## Completed checks

- Full test suite: **232 passed** in 21.90 seconds, including real raster reprojection, API persistence, observation imports, monitoring analysis, historical evaluation, model contracts, and saved plots.
- Frontend syntax: `node --check forestmind/web/app.js` passed.
- Browser: checked region selection, alerts and observations, JSON import validation, scenario calculation and invalidation after input changes, historical backtesting, and responsive layouts. No browser console errors were observed in the final preview.
- CPU training and evaluation: the small synthetic configuration completed and produced `checkpoints/quick_test/best_model.pt`.
- Checkpoint demo: completed successfully with the configured optical input shape of 4 time steps by 12 bands by 32 by 32 pixels.
- Packaging: built `artifacts/dist/forestmind-0.3.0-py3-none-any.whl`, including the dashboard assets and API.

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

## Research limits

These checks establish software behavior, not ecological predictive validity. The seeded monitoring regions and quick training run use synthetic data. Native pretrained checkpoints, live satellite ingestion, and operational deployment were not exercised. Monitoring scores and scenario outputs remain uncalibrated research indicators. Validation on independent real observations and events, geographic and temporal holdouts, and uncertainty calibration remain necessary before scientific or operational claims.

See `METHODOLOGY.md` and `RESEARCH_PROTOCOL.md` for the implemented assumptions and evaluation requirements.

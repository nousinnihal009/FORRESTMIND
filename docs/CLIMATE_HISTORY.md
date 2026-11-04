# Reproducible real climate history

The climate history builder obtains bounded ERA5 windows from the existing fixed Open-Meteo archive endpoint and assembles the [frozen reference diagnostics](CLIMATE_BASELINE.md). It retains exact raw responses, dates, provider metadata, hashes, missingness and centroid/grid identities. It does not write to the monitoring database.

```powershell
python scripts/build_climate_history.py --area docs/pilots/mudumalai.json --output data/research-climate-mudumalai-20261105
python scripts/build_climate_history.py --area docs/pilots/mudumalai.json --output data/research-climate-mudumalai-20261105 --check
```

Defaults freeze 1991–2020 from three non-overlapping ten-year requests, then analyze separate 2021–2025 observations. A new directory is required initially. Use the same command with `--resume` to verify cached sources and continue missing windows within the recorded attempt budgets. Each call is bounded by `--max-requests`; the default four requests are spaced by 35 seconds. Permanent denial/rate limiting stops further network requests. The public hosted service's [usage conditions and variable definitions](https://open-meteo.com/en/docs/historical-weather-api) still apply; the tool does not purchase an API plan.

`plan.json` pins the geometry, periods, implementation and budgets. `progress.json` records requests and accepted file hashes. Accepted files are never replaced. All cached files are verified before a new request. An interrupted request retains its attempt count; a source saved before a progress update can be recovered only after matching its contract. Simultaneous builds are refused with a local lock; after an abruptly killed process, confirm the owning process has stopped before removing its stale lock.

`--check` verifies a completed packet without writes or requests, including recomputing the reference and departures. It can verify evidence obtained by an older pinned builder without rewriting its provenance. Resume still requires the original implementation. The completed real November 5 packet predates a bounded Windows metadata-write retry fix; its exact builder is preserved at `artifacts/climate-history-builder-used-20261105.py`, matching the fingerprint in its plan. Use `--check` for that completed packet with current source. Temporary Windows sharing failures retry only atomic progress replacement, at most five times; persistent errors preserve the old progress record and never trigger another download.

When all windows pass, `reference.json` and `departures.json` are written as immutable hashed artifacts. Completion means source and calculation contracts passed, not ecological validity. At least 30 complete reference years per calendar-month/feature pair are required; absent measurements remain missing. No climatological period is selected based on forest test performance.

The actual Mudumalai build completed four requests: 1991–2000, 2001–2010, 2011–2020 and 2021–2025. All 48 reference feature/month pairs and 240 observed feature/month departures across 60 months are available. The sources are real reanalysis, not test fixtures. Thirty-five years of climate do not create 35 years of optical imagery or independent forest-outcome labels. Historical reanalysis vintage availability remains unverified; the outputs remain descriptive context rather than neural inputs or validated drought/tipping predictions.

Implementation: `forestmind/data/climate_history.py`, `scripts/build_climate_history.py`. Tests exercise plan bounds, missingness, persistent budgets, resume integrity, source mismatches, storage failure and immutable derived records. Actual execution evidence is recorded in `VALIDATION.md`.

# Frozen climate references and monthly departures

The local climate diagnostic tools compare observed monthly ERA5 measurements with a separately frozen, earlier reference. They calculate reproducible standardized departures; they do **not** fit SPI/SPEI, diagnose ecological drought, train a predictor, or establish a tipping point. Existing four-feature neural climate inputs, the dashboard, the database and saved source observations are unchanged by these tools.

## Required evidence

Inputs are existing ForestMind Open-Meteo ERA5 context JSON files containing `raw_response.daily`, the original units, requested coordinates, returned grid point, retrieval time, request URL and provider-response hash. A selected-area analysis export containing this same `climate` object also works. Files are read locally; the commands make no provider request.

Each context may contain at most ten years / 3,660 daily records. Up to twelve disjoint context files can be combined. The declared reference must span 30–60 complete calendar years, and **every calendar-month/feature pair requires at least 30 complete yearly observations**. For example, January rainfall needs 30 distinct reference Januaries with all daily precipitation measurements present. One incomplete January excludes that year from January rainfall; it does not erase a complete January temperature record. Insufficient pairs retain counts and excluded years but have no fitted mean, standard deviation or departure.

The 30-year eligibility rule is conservative software policy informed by the [WMO climatological-normal duration](https://community.wmo.int/site/knowledge-hub/programmes-and-initiatives/climate-services/wmo-climatological-normals). These custom, strict-completeness calculations are **not certified WMO normals**. They do not implement all WMO methods, homogenization or missing-data guidance. Complete-case selection can be biased when source gaps are systematic. The actual accepted and excluded years need scientific review.

All contexts must identify the same provider, dataset, requested centroid, returned grid point and elevation, units, spatial-resolution declaration and license. UTC/GMT dates and original variable definitions are mandatory. Conflicting or overlapping context windows, duplicate or unordered daily dates, invalid numbers, negative precipitation/reference ET₀, out-of-range volumetric moisture and synthetic declarations are rejected. Hashes establish local consistency, not independent provider authenticity.

## Calculations and limits

For each complete reference year and calendar month, the implementation stores:

| Feature | Monthly aggregation | Unit |
| --- | --- | --- |
| Temperature | Mean of daily mean 2 m temperature | °C |
| Rainfall | Sum of daily precipitation | mm |
| Surface soil moisture | Mean of daily 0–7 cm volumetric soil moisture | m³/m³ |
| Climatic balance | Sum of daily precipitation minus reference ET₀, requiring both inputs on every day | mm |

The reference stores every accepted annual aggregate. Its mean and **sample** standard deviation (`ddof=1`) are calculated across the same calendar month's accepted years. A later complete month has physical departure `observed − reference_mean` and standardized departure `(observed − reference_mean) / reference_sample_sd`. A zero reference standard deviation leaves the standardized value null while retaining the physical departure. No distributional normality or calibrated tail probability is claimed.

The `warm_month` diagnostic compares **monthly mean daily temperature** strictly with the reference monthly 90th percentile. Quantiles use linear interpolation at `(n−1)q` in the sorted yearly aggregates. This is a descriptive warm-month comparison, not a daily maximum-temperature extreme, heatwave threshold or ecological stress diagnosis. Equal-to-threshold values are not exceedances.

Reference ET₀ represents atmospheric demand for a reference grass surface. Consequently P−ET₀ is a climatic balance, not actual forest water use or an ecosystem water budget. Values are reanalysis at a centroid/grid point, not field observations. See the [Open-Meteo historical variable documentation](https://open-meteo.com/en/docs/historical-weather-api).

Every observation window must start **after** the frozen reference ends, and end on or before the explicitly supplied cutoff. An observation source containing dates after the cutoff is rejected; it is not silently cropped. Analysis covers at most ten years, from the first supplied observed context through the cutoff. Gaps between supplied contexts and unobserved months through the cutoff remain visible.

Only complete calendar months receive departures. Partial first/last months, absent days and null measurements retain separate expected-day, valid-day, unreported-day and missing-measurement counts. Rainfall of zero remains an actual measurement. No partial sum is rescaled. Leap days use the actual calendar. Temperature, rainfall and soil moisture have independent coverage; climatic balance requires same-day rainfall/ET₀ pairs.

Date separation alone does not establish that a given ERA5 revision was available at a historical forecast origin. `historical_availability_verified` remains false. A study must audit historical availability or disclose retrospective reanalysis. These standalone results are marked `model_input: false`; adding them to a predictor requires a separately versioned feature contract, training-only transformations and evaluation.

## Local commands

The filenames below are examples for **already obtained** source evidence. They are not bundled datasets. For post-2020 observations, a 1991–2020 reference can be frozen from three non-overlapping ten-year contexts:

```powershell
python scripts/climate_diagnostics.py freeze --context data/era5-1991-2000.json --context data/era5-2001-2010.json --context data/era5-2011-2020.json --baseline-start 1991-01-01 --baseline-end 2020-12-31 --output artifacts/era5-reference-1991-2020.json
python scripts/climate_diagnostics.py analyze --reference artifacts/era5-reference-1991-2020.json --context data/era5-observed-2021.json --cutoff 2021-12-31 --output artifacts/climate-departures-2021.json
```

Use an earlier reference period when the study's observation cutoff precedes 2021; never include a target period in its reference. The tools do not choose a reference period based on predictive performance. Read source usage rights separately: accepting a recorded license string does not verify legal access.

The CLI accepts at most twelve source files, each at most 16 MiB. Duplicate JSON keys and nonfinite JSON literals are rejected. Outputs are new files, never replacements. Both commands return exit status 2 on invalid inputs or an existing output path. A successfully written reference can still contain unavailable feature/month pairs; inspect `available_feature_months`, counts and reasons. A successful command does not imply sufficient observations or scientific validity.

Artifacts preserve source-context and canonical raw-response hashes, the supplied original provider-response hash, exact file-byte hashes, reference years, per-year aggregates, methods, units, missingness and limitations. The final canonical artifact hash covers these fields. Loading a reference rechecks the digest, recalculates its summary statistics from the stored annual aggregates, and rejects sample months outside the declared source windows. Retain the source files: summary hashes do not replace raw evidence or independently authenticate measurements. An interrupted/incomplete JSON file is not a valid reusable reference.

## Verification and current evidence

Implementation: `forestmind/monitoring/climatology.py`; CLI: `scripts/climate_diagnostics.py`; focused contracts: `tests/test_climatology.py`.

Tests exercise analytic means/standard deviations, frozen-reference isolation, same-source/grid validation, missingness by variable, leap years, zero rainfall, zero variance, gaps, strict cutoffs, corrupted/rehashed records, JSON bounds, immutable file writes and CLI round-trips. Fixtures are explicitly generated software-test records; they provide no real ecological or climate validation.

The saved Mudumalai export was read without changes. Its ERA5 context contains **323 daily rows from 12 October 2025 through 30 August 2026** and passes the existing-source contract check. Its short history is correctly rejected as a 30-year reference. No new historical climate data or populated real-reference anomaly results were produced. A suitable frozen historical source dataset remains the dependency for actual use.

# Methods, units and interpretation

## Execution paths

The observation dashboard uses `transparent-indicator-baseline-v1`. Connected analysis can also run an eligible registered neural checkpoint with real-training declarations, independent evaluation and a compatible input/geographic contract. No such checkpoint is bundled. The research training pipeline requires independent labels; controlled synthetic tests do not approve a model for real observations. Neither path establishes a calibrated ecological tipping probability.

## Observations

Each row represents at most one calendar month and one AOI. The cache converter uses quality-masked finite reflectance in `[0,1]`, excludes near-zero denominators, computes per-pixel indices, then takes a valid-pixel-sample-weighted monthly mean across acquisitions. It does not calculate an index from the monthly mean reflectance.

| Index | Formula | Meaning |
| --- | --- | --- |
| NDVI | `(NIR-red)/(NIR+red)` | Greenness-related proxy |
| NDMI | `(NIR-SWIR1)/(NIR+SWIR1)` | Moisture-related proxy |
| NBR | `(NIR-SWIR2)/(NIR+SWIR2)` | Disturbance-sensitive proxy, not proof of fire |

Valid fraction counts usable pixel samples divided by all pixel samples across acquisitions; it is not a union of clear geographic coverage. Clouds, mixed land cover, bandpass differences and atmosphere remain confounders. The converter summarizes the AOI, not an independently verified forest mask. Month labels use the latest actual UTC acquisition date; gaps remain gaps.

Rasterio WarpedVRT aligns every band onto an explicit WGS84 grid. Continuous bands use bilinear interpolation; QA uses nearest neighbor. Transform and angular pixel size are recorded. Native sensor resolution and output resolution differ. Legacy caches must be regenerated for strict import preparation.

## Condition index

All three indices and valid fraction >=0.4 are required:

```text
v = clip((NDVI - 0.2) / 0.7, 0, 1)
m = clip((NDMI + 0.1) / 0.7, 0, 1)
b = clip(NBR / 0.8, 0, 1)
score = 100 × (0.40 v + 0.35 m + 0.25 b)
```

This inspectable mapping is not field calibrated, universal across biomes, or a percent of forest remaining. Status thresholds are stable >=75, watch >=58, stressed >=40, otherwise critical. Unusable latest measurements produce insufficient_data. The word stable does not establish ecological stability.

Displayed change compares the latest three records with the preceding twelve, using available scores within those windows. It is not single-month change or a seasonal significance test. Coverage is the latest three-record mean valid fraction. Workspace mean condition is an unweighted mean across regions. Alerts flag index stress, a decline of at least five points, and quality gaps for investigation.

## Driver evidence

Rainfall deficits, temperature anomalies, supplied fire anomalies, canopy-cover differences, radar changes and NBR changes produce separate bounded evidence strengths. Scores are not causal probabilities or percentages of contribution. Multiple drivers may coexist. Radar responds to moisture and viewing geometry; NBR alone cannot establish fire or logging. Missing evidence stays unavailable.

Context units are precipitation/fire anomalies in standard deviations against a documented reference, heat anomalies in °C, SAR change in dB and forest cover in `[0,1]`. The importer checks types/ranges; it cannot prove source accuracy or compatible anomaly baselines.

## Forecast baseline

Requires a usable latest observation and twelve consecutive usable months. Fit a linear slope against elapsed days over the latest twelve usable records. At horizon `h`, `y=h/365.25`:

```text
forecast = clip(current + slope_per_year × (1-exp(-y)) × 0.65, 0, 100)
spread = max(residual standard deviation, 2.5)
envelope = 1.96 × spread × sqrt(1+h/90)
```

Bounds are clipped to `[0,100]`. The 1.96 multiplier does **not** establish 95% coverage. The damping and envelope are uncalibrated. Future fires, policy decisions and new disturbances are not predicted. A twelve-month slope can confound seasonality and secular change; persistence/seasonal-naive backtesting helps inspect that limitation.

## Resilience

The dashboard requires 36 consecutive usable monthly scores. It removes linear trend and month-of-year means, compares the latest and previous eighteen-month residual windows, and flags a joint increase when autocorrelation rises by >0.1 and variance exceeds 1.25 times the prior window. These are exploratory thresholds. Constant residuals yield unavailable correlation; absence of an increase does not prove resilience.

The separate research risk module accepts explicit timing and seasonal settings. Neither module invents spatial fragmentation, drought response, tipping dates or calibrated confidence from a health curve. Landscape connectivity needs a spatial forest mask; drought response needs aligned climate evidence.

## Scenarios

Dashboard sliders in `[0,100]` produce a declared annual condition-point shift:

```text
shift = 0.04×fire_reduction + 0.05×pressure_reduction
      + 0.03×restoration - 0.08×climate_stress
```

Horizon effects scale by `h/365`; zero changes reproduce the baseline. Coefficients are assumptions, not estimated treatment effects or management recommendations. Results retain parameters, baseline, provenance and limitations. Research intervention modules likewise expose assumed reductions, delays and configurable horizons.

## Unestablished claims

Field-calibrated health states, validated degradation probabilities, causal disturbance attribution, calibrated uncertainty, incremental benefit from real pretrained representations, five-year forecasts, irreversible tipping-point prediction and causal intervention effects remain research objectives. Software correctness alone does not establish these claims.

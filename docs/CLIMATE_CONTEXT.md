# Climate context and paired climatic balance

The **Collect climate context** action requests actual ERA5 reanalysis for the selected area's centroid and the accepted observation window. It retains daily provider values, units, requested and returned coordinates, dates, retrieval time and the raw-response SHA256. This is a coarse model-based record, not a forest-only measurement. Original daily values can supply lagged trailing features to a compatible registered neural model with explicit feature masks, normalization and coverage; the monthly table itself is descriptive. Historical reanalysis vintage availability remains unverified. See [MULTIMODAL_INPUTS.md](MULTIMODAL_INPUTS.md). Provider details and terms are in the [Open-Meteo historical API documentation](https://open-meteo.com/en/docs/historical-weather-api).

## Paired precipitation minus reference evapotranspiration

For each requested calendar-month interval, method `paired-daily-precipitation-minus-reference-et0-v1` calculates:

`sum(P[d] − ET0[d])`, over only dates on which both source values are finite and present.

The result is in millimetres. The exported `mean_daily_mm` divides this sum by the number of paired days and has units mm/day. Each row retains the paired dates, paired-day count, expected requested-day count, missing-pair count, unreported-day count and exact interval. Zero precipitation and zero reference evapotranspiration remain valid values. No paired dates produces null, not zero.

This paired sum may differ from subtracting the two displayed monthly source sums: their valid days can differ. No values are filled, and no unpaired input contributes. Whole requested months with no returned dates appear with null summaries and explicit missing counts. Per-variable `missing_days` counts nulls within returned dates; `missing_requested_days` also counts dates absent from the response.

The display distinguishes a complete calendar month, a fully observed partial-month request and a partial sum with missing daily pairs. A partial sum is not comparable with a complete-month total. Leap days are counted from the requested calendar; no fixed 30-day denominator is used.

Reference ET₀ describes the atmospheric water demand of a well-watered grass reference surface; it does not measure actual forest evapotranspiration. Therefore P − ET₀ is a descriptive climatic balance, not an ecosystem water budget, a standardized drought index, a soil-water deficit or proof of vegetation stress. No drought threshold, climatology or causal driver score is inferred. See the provider's [ET₀ definition](https://open-meteo.com/en/docs/historical-weather-api).

## Saved evidence and limitations

**Run selected-area analysis** can derive this diagnostic from already saved daily context without another provider request. It saves a copy of the context with the new summaries and creates a new analysis manifest. Original daily values, raw-response hash, retrieval timestamp, geometry and cutoff remain unchanged; historical snapshots and their exports remain immutable. Legacy context without daily evidence is left unchanged because same-day pairing cannot be reconstructed safely from monthly totals.

ERA5 centroid sampling, resolution, latency and reanalysis uncertainty still apply. These columns do not add radar/climate inputs to an optical model. Seasonal anomaly and heat-extreme definitions, a justified historical climatology, standardized drought diagnostics and validation against independent ecological observations remain separate unfinished research work.

Software tests cover disjoint missing days, genuine zeroes, leap and partial months, wholly unreported months, nonfinite inputs, immutable raw evidence, provider-free rederivation and unchanged historical snapshots. These checks establish computation and provenance behavior, not ecological predictive validity.

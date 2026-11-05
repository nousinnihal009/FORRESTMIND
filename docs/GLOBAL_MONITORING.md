# Worldwide forest monitoring

ForestMind's product scope is forests worldwide. Mudumalai is one development and data-acquisition pilot, alongside other saved test areas; it does not set a country restriction, training population or deployment boundary. The original proposal's advice to begin with one region is an experimental staging decision. It does not limit the final application to that region.

## One workflow, separate evidence for each location

Use **Find a forest** for place search, map selection or coordinates. A named place is a location reference, not a verified forest boundary. The optical, radar, climate and classified-cover workflows use the selected geometry, dates and accepted sources. The long-term climate-history CLI also accepts an arbitrary area file; its saved Mudumalai run is an example, not its only supported location.

For a larger study, use **Plan a larger forest**. Save a named bounding rectangle, review the component monitoring areas, and choose one to prepare in the existing acquisition form. Confirm its exact boundary, dates and collection budget there. Planning saves metadata only. It does not download imagery or queue collection across an entire continent.

Plans support both hemispheres and longitude crossings at the international date line. In a crossing rectangle, west longitude is greater than east longitude. The planner splits it into ordinary rectangles on either side of 180°. Supported latitude bounds remain 85°S–85°N. Large plans are paginated rather than expanded into millions of database records or jobs. Each monitoring area retains the existing maximum of 25 km² and a maximum side span of 1°.

Saving a plan preserves its geometry and grid method. The same inputs reproduce its monitoring areas. Observations collected for an exact matching rectangular footprint appear with that area; a partial overlap, a polygon with the same bounding box, or a synthetic demonstration does not establish observed coverage. A region containing some measurements still needs its own quality and time-coverage checks.

## What coverage means

The plan counts component areas with saved nonsynthetic observations. This is an inventory of available records, not the percentage of the forest that is healthy, the fraction of every pixel observed, or the completeness of the requested time series. Missing areas stay missing. Open each area's Data coverage and Connected analysis views to inspect valid pixels, dates, gaps, source compatibility and supported analyses.

Rectangular plans can include nonforest land. Verified forest boundaries, forest masks and area-based reference measurements remain separate evidence. Tree-cover classes do not establish ecological condition. Do not average condition indices across biomes into a global health score, or average independent tile fragmentation statistics into whole-forest connectivity: tile edges can divide real patches. A worldwide map is also not a globally validated prediction model.

## Global scientific acceptance

The same observation → diagnosis → forecasting → scenario workflow applies everywhere, but its ecological interpretation must be supported locally. Calendar-month baselines should come from each area's own history. Drought thresholds, seasonal behavior, reference definitions and model applicability cannot be inferred from Mudumalai or transferred automatically from a tropical forest to a boreal forest. The existing transparent condition score remains a heuristic, not a universal ecological classification.

Models continue to require a compatible checkpoint, real independent evaluation, the documented geographic/input scope and sufficient observations. Larger study plans do not bypass these checks. Cross-region acceptance needs protected geographic and temporal tests, biome and seasonal subgroup reporting, calibration on separate evidence, and useful warning performance. The saved global public-reference experiment is retrospective point-level evidence; it neither validates the primary 100 ha target nor establishes worldwide early-warning skill.

## Capacity and remaining scope

The local service keeps bounded storage, one acquisition worker and explicit download budgets. Plans let a large forest be organized without removing those limits. Acquisition and analysis remain per area; unattended collection of every planned area, a verified inventory of every forest, whole-landscape spatial mosaics, hosted multi-user operation and worldwide validated forecasting are not supplied by this planner. Provider coverage and data quality still determine what can be observed at a particular location and date.

Saved plans live in the workspace SQLite database and travel with database snapshots and full workspace backups. No new external account or paid service is required. See [ACQUISITION_OPERATIONS.md](ACQUISITION_OPERATIONS.md) for collection limits and [REQUIREMENTS_MATRIX.md](REQUIREMENTS_MATRIX.md) for the remaining scientific and software requirements.

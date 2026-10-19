"""Measured spatial diagnostics; no forest-cover labels inferred from greenness."""

from __future__ import annotations

from datetime import date
import math

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree


def geometry_bbox(geometry: dict) -> list[float]:
    points = geometry["coordinates"][0]
    return [min(p[0] for p in points), min(p[1] for p in points),
            max(p[0] for p in points), max(p[1] for p in points)]


def polygon_mask(geometry: dict, shape: tuple[int, int]) -> np.ndarray:
    """Pixel-centre inclusion on the exact selected WGS84 grid."""
    west, south, east, north = geometry_bbox(geometry)
    height, width = shape
    x, y = np.meshgrid(west + (np.arange(width) + .5) * (east - west) / width,
                       north - (np.arange(height) + .5) * (north - south) / height)
    inside = np.zeros(shape, dtype=bool)
    for a, b in zip(geometry["coordinates"][0], geometry["coordinates"][0][1:]):
        if a[1] == b[1]:
            continue
        crosses = ((a[1] > y) != (b[1] > y))
        inside ^= crosses & (x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0])
    return inside


def spectral_change(prepared, geometry: dict) -> dict:
    """First/latest common-valid-pixel spectral deltas, never forecast risk."""
    arrays = prepared.arrays
    data = arrays["optical"]
    dates = arrays["acquisition_dates"].astype(str).tolist()
    if len(data) < 2:
        return {"available": False, "reason": "Two accepted, quality-masked optical acquisitions are required; one is available."}
    start, end = data[0], data[-1]
    valid = arrays["quality_mask"][0] & arrays["quality_mask"][-1] & polygon_mask(geometry, start.shape[1:])
    outputs = {}
    for name, other in (("ndvi", 3), ("ndmi", 10), ("nbr", 11)):
        den0, den1 = start[7] + start[other], end[7] + end[other]
        good = valid & (den0 > 1e-8) & (den1 > 1e-8)
        before = np.divide(start[7] - start[other], den0, out=np.zeros_like(den0), where=good)
        after = np.divide(end[7] - end[other], den1, out=np.zeros_like(den1), where=good)
        delta = after - before
        outputs[name] = {"values": [[round(float(delta[y, x]), 5) if good[y, x] else None
                                      for x in range(delta.shape[1])] for y in range(delta.shape[0])],
                         "mean_delta": float(delta[good].mean()) if good.any() else None,
                         "valid_fraction": float(good.sum() / max(1, polygon_mask(geometry, good.shape).sum()))}
    return {"available": bool(valid.any()), "kind": "observed_spectral_change", "units": "index difference (latest minus first)",
            "reason": "Actual common-valid pixels only." if valid.any() else "No common valid pixels across the two dates.",
            "start": dates[0], "end": dates[-1], "bbox": geometry_bbox(geometry),
            "shape": list(start.shape[1:]), "crs": "EPSG:4326", "layers": outputs,
            "sources": [prepared.provenance["sources"][0], prepared.provenance["sources"][-1]],
            "limitations": ["Observed spectral change is not forest loss, burn severity, or future degradation probability.",
                            "Different seasons, illumination, sensor processing and remaining cloud effects can explain change.",
                            "Grid resolution is AOI extent / raster dimensions; resampling creates no new detail."]}


def recovery_diagnostics(observations: list[dict], threshold: float = .1) -> dict:
    """Seasonal NDVI anomaly episodes with a frozen, past-only two-year baseline.

    Recovery requires two consecutive monthly observations within half the
    episode threshold. Missing months right-censor episodes; no interpolation.
    """
    usable = [r for r in observations if r.get("ndvi") is not None and r["valid_fraction"] >= .4]
    base = {"available": False, "method": "fixed-two-year-seasonal-ndvi-recovery-v1",
            "threshold_ndvi": threshold, "episodes": [],
            "interpretation": "Observed return toward historical NDVI, not ecological recovery, critical slowing down, or a tipping time."}
    if len(usable) < 27:
        return {**base, "reason": f"Need 24 consecutive usable baseline months and at least 3 subsequent months; {len(usable)} usable observations available."}
    def month(row):
        value = date.fromisoformat(row["date"])
        return value.year * 12 + value.month
    if any(month(b) - month(a) != 1 for a, b in zip(usable[:24], usable[1:24])):
        return {**base, "reason": "The first 24 usable observations contain missing months; a contiguous seasonal baseline is required."}
    baseline = {m: float(np.median([r["ndvi"] for r in usable[:24] if date.fromisoformat(r["date"]).month == m]))
                for m in range(1, 13)}
    episodes, active, recovered = [], None, 0
    previous = usable[23]
    for row in usable[24:]:
        if month(row) - month(previous) != 1 and active:
            active.update({"status": "censored_missing_months", "last_observed": previous["date"], "recovery_days": None})
            episodes.append(active)
            active, recovered = None, 0
        anomaly = row["ndvi"] - baseline[date.fromisoformat(row["date"]).month]
        if active is None and anomaly <= -threshold:
            active = {"start": row["date"], "minimum_anomaly": float(anomaly)}
            recovered = 0
        if active:
            active["minimum_anomaly"] = min(active["minimum_anomaly"], float(anomaly))
            recovered = recovered + 1 if anomaly >= -threshold / 2 else 0
            if recovered == 2:
                active.update({"status": "observed_return", "recovered_at": row["date"],
                               "recovery_days": (date.fromisoformat(row["date"]) - date.fromisoformat(active["start"])).days})
                episodes.append(active)
                active, recovered = None, 0
        previous = row
    if active:
        active.update({"status": "right_censored", "last_observed": usable[-1]["date"], "recovery_days": None})
        episodes.append(active)
    return {**base, "available": True, "reason": "No qualifying negative NDVI episodes detected." if not episodes else "Episodes include unresolved and missing-observation censoring.",
            "baseline_start": usable[0]["date"], "baseline_end": usable[23]["date"],
            "baseline_by_month": baseline, "episodes": episodes,
            "limitations": ["Two years provide a short climatology; thresholds are exploratory and uncalibrated.",
                            "Comparisons across episodes require independent disturbance severity and climate controls."]}


def fragmentation_metrics(payload: dict, geometry: dict) -> dict:
    """4-connected forest patch topology from a separately supplied cover raster."""
    cover = np.asarray(payload["forest"], dtype=bool)
    valid = np.asarray(payload["valid_mask"], dtype=bool) & polygon_mask(geometry, cover.shape)
    forest = cover & valid
    patches, count = label(forest, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    west, south, east, north = geometry_bbox(geometry)
    height, width = forest.shape
    dy = math.radians(north - south) * 6371008.8 / height
    latitudes = north - (np.arange(height) + .5) * (north - south) / height
    dx = math.radians(east - west) * 6371008.8 * np.cos(np.radians(latitudes)) / width
    areas = np.broadcast_to(dx[:, None] * dy, forest.shape)
    landscape_area = float(areas[valid].sum())
    patch_areas = np.bincount(patches.ravel(), weights=areas.ravel(), minlength=count + 1)[1:]
    horizontal = valid[:, :-1] & valid[:, 1:] & (forest[:, :-1] != forest[:, 1:])
    vertical = valid[:-1] & valid[1:] & (forest[:-1] != forest[1:])
    edge = float(horizontal.sum() * dy + (vertical * ((dx[:-1] + dx[1:]) / 2)[:, None]).sum())
    centres = []
    for index in range(1, count + 1):
        yy, xx = np.nonzero(patches == index)
        centres.append([float((xx * dx[yy]).mean()), float((yy * dy).mean())])
    nearest = cKDTree(centres).query(centres, k=2)[0][:, 1] if count > 1 else np.array([])
    forest_area = float(areas[forest].sum())
    return {"available": True, "kind": "forest_cover_topology", "source": payload["source"],
            "observed_at": payload["observed_at"], "classification": payload["classification"],
            "provider": payload.get("provider", "User-supplied classification"), "license": payload.get("license"),
            "published_at": payload.get("published_at"), "retrieved_at": payload.get("retrieved_at"),
            "temporal_role": payload.get("temporal_role", "Dated user-supplied reference; availability history is not independently verified"),
            "valid_fraction": float(valid.mean()), "patch_count": int(count), "connectivity_rule": "four-neighbour adjacency",
            "forest_area_km2": forest_area / 1e6, "valid_area_km2": landscape_area / 1e6,
            "forest_fraction": forest_area / landscape_area if landscape_area else None,
            "patch_area_km2": (patch_areas / 1e6).tolist(), "internal_edge_km": edge / 1000,
            "edge_density_m_per_ha": edge / (landscape_area / 10000) if landscape_area else None,
            "largest_patch_fraction": float(patch_areas.max() / forest_area) if forest_area else None,
            "mean_nearest_centroid_m": float(nearest.mean()) if len(nearest) else None,
            "limitations": ["Reference classification has not been independently field-validated for this AOI.",
                            "AOI boundary and nodata boundaries are excluded from edge length; outside connectivity is unknown.",
                            "Four-neighbour topology and centroid distance do not measure species-specific habitat connectivity.",
                            "Areas and distances use a spherical local-grid approximation. No forest class is inferred from vegetation indices.",
                            *payload.get("limitations", [])]}

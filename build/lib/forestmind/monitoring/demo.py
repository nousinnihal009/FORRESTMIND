"""Reproducible fictional observations over four illustrative Amazon AOIs."""

from __future__ import annotations

import math
import numpy as np


def demo_regions() -> list[tuple[dict, list[dict]]]:
    definitions = [
        ("tapajos", "Tapajós National Forest", [-55.02, -3.36], 5274, 0.0, 0.07, 107),
        ("xingu", "Xingu Basin", [-52.69, -8.85], 14280, 0.24, 0.25, 208),
        ("madeira", "Madeira Corridor", [-63.51, -8.14], 8640, 0.16, 0.18, 309),
        ("rio-negro", "Rio Negro Reserve", [-61.92, -1.51], 11620, 0.0, 0.01, 410),
    ]
    output = []
    for region_id, name, center, area, initial_stress, decline, seed in definitions:
        lon, lat = center
        # Rectangles are illustrative AOIs, not legal protected-area boundaries.
        dx = math.sqrt(area) / (2 * 111.32 * math.cos(math.radians(lat)))
        dy = math.sqrt(area) / (2 * 111.32)
        ring = [[lon - dx, lat - dy], [lon + dx, lat - dy], [lon + dx, lat + dy],
                [lon - dx, lat + dy], [lon - dx, lat - dy]]
        region = {
            "id": region_id, "name": name, "country": "Brazil", "center": center,
            "area_km2": area, "geometry": {"type": "Polygon", "coordinates": [ring]},
            "provenance": {
                "mode": "synthetic_demo", "synthetic": True,
                "source": "Deterministic fictional monthly series (seeded simulation)",
                "method": "transparent-indicator-baseline-v1", "model": "No trained model used",
                "period": "2022-10 through 2026-09", "seed": seed,
                "geometry_note": "Illustrative AOI rectangle, not an official administrative boundary.",
                "verified": False,
            },
        }
        rng = np.random.default_rng(seed)
        rows = []
        for index in range(48):
            month_absolute = 2022 * 12 + 9 + index
            year, month_zero = divmod(month_absolute, 12)
            season = math.sin(month_zero / 12 * 2 * math.pi)
            stress = initial_stress + decline * (index / 47) ** 1.6
            rows.append({
                "date": f"{year:04d}-{month_zero + 1:02d}-01",
                "ndvi": round(float(np.clip(0.86 - stress * 0.65 + 0.025 * season + rng.normal(0, 0.01), -1, 1)), 4),
                "ndmi": round(float(np.clip(0.54 - stress * 0.75 + 0.035 * season + rng.normal(0, 0.015), -1, 1)), 4),
                "nbr": round(float(np.clip(0.77 - stress * 0.85 + 0.025 * season + rng.normal(0, 0.012), -1, 1)), 4),
                "precipitation_anomaly": round(float(-stress * 4 + season * 0.35 + rng.normal(0, 0.15)), 3),
                "temperature_anomaly": round(float(0.2 + stress * 5 + rng.normal(0, 0.12)), 3),
                "sar_change": round(float(-stress * 5 + rng.normal(0, 0.15)), 3),
                "forest_cover": round(float(np.clip(0.97 - stress * 0.38, 0, 1)), 4),
                "fire_anomaly": round(float(max(0, stress * 5 + rng.normal(0, 0.2))), 3),
                "valid_fraction": round(float(rng.uniform(0.74, 0.98)), 3),
            })
        output.append((region, rows))
    return output

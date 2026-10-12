"""Band alignment for Landsat and Sentinel-2 observations.

This aligns broadly corresponding bands; it is NOT the NASA HLS processing
chain and does not perform atmospheric, BRDF, or spectral-response corrections.
Missing red-edge/water-vapour measurements are zero-filled by default. Optional
interpolation is an explicitly synthetic approximation, not an observation.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# Direct band placement only. No unverified HLS correction coefficients.
BAND_MAPPING = {"B01": 0, "B02": 1, "B03": 2, "B04": 3,
                "B08": 4, "B11": 5, "B12": 6}
# Compatibility alias: identity mappings, not published HLS coefficients.
HLS_COEFFICIENTS = {band: (index, 1.0, 0.0) for band, index in BAND_MAPPING.items()}

# Target S2 band order (matches our 12-band convention)
S2_BAND_ORDER = [
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B11", "B12",
]


class BandHarmonizer:
    """Cross-sensor spectral alignment module.

    Transforms multi-sensor observations into a unified 12-band
    Sentinel-2-equivalent representation. Enables multi-source
    time series to be processed by a single model.

    Supports:
      - Landsat-8/9 (7 bands) → Sentinel-2 (12 bands)
      - Sentinel-2 pass-through (already 12 bands)

    Args:
        target_bands: Target band names (default: 12 S2 bands).
        interpolate_red_edge: Whether to estimate red-edge bands from
            Landsat data or fill with zeros.
    """

    def __init__(
        self,
        target_bands: Optional[list[str]] = None,
        interpolate_red_edge: bool = False,
    ):
        self.target_bands = target_bands or S2_BAND_ORDER
        if any(band not in S2_BAND_ORDER for band in self.target_bands):
            raise ValueError("Unknown Sentinel-2 target band")
        self.interpolate_red_edge = interpolate_red_edge
        self._build_transform_matrix()

    def _build_transform_matrix(self) -> None:
        """Pre-compute the transformation matrix for Landsat → S2."""
        self.landsat_slopes = np.zeros(len(self.target_bands), dtype=np.float32)
        self.landsat_offsets = np.zeros(len(self.target_bands), dtype=np.float32)
        self.landsat_source_idx = np.zeros(len(self.target_bands), dtype=np.int32)

        for i, band in enumerate(self.target_bands):
            if band in HLS_COEFFICIENTS:
                src_idx, slope, offset = HLS_COEFFICIENTS[band]
                self.landsat_source_idx[i] = src_idx
                self.landsat_slopes[i] = slope
                self.landsat_offsets[i] = offset

    def harmonize_landsat(self, patch: np.ndarray) -> np.ndarray:
        """Transform a Landsat 7-band patch to 12-band S2 equivalent.

        Uses direct band placement, with optional synthetic red-edge interpolation.

        Args:
            patch: (7, H, W) Landsat surface reflectance in [0, 1].

        Returns:
            (12, H, W) Sentinel-2-equivalent surface reflectance.
        """
        if patch.shape[0] != 7:
            raise ValueError(f"Expected 7-band Landsat input, got {patch.shape[0]} bands")

        _, H, W = patch.shape
        output = np.zeros((len(self.target_bands), H, W), dtype=np.float32)

        for i, band in enumerate(self.target_bands):
            if band in HLS_COEFFICIENTS:
                src_idx, slope, offset = HLS_COEFFICIENTS[band]
                output[i] = patch[src_idx] * slope + offset
            else:
                # Should not happen with current config, but handle gracefully
                output[i] = 0.0

        if self.interpolate_red_edge:
            # Explicitly synthetic red-edge approximation between red and NIR.
            for band, weight in (("B05", 0.25), ("B06", 0.5), ("B07", 0.75), ("B8A", 1.0)):
                if band in self.target_bands:
                    output[self.target_bands.index(band)] = (1 - weight) * patch[3] + weight * patch[4]
        # If not interpolating red-edge, zero those bands
        if not self.interpolate_red_edge:
            for i, band in enumerate(self.target_bands):
                if band in ("B05", "B06", "B07", "B8A", "B09"):
                    output[i] = 0.0

        # Clip to valid range
        output = np.clip(output, 0.0, 1.0)

        return output

    def harmonize(
        self,
        patch: np.ndarray,
        source_sensor: str,
    ) -> np.ndarray:
        """Harmonize a patch from any supported sensor to S2-equivalent.

        Args:
            patch: (C, H, W) input patch.
            source_sensor: Sensor identifier ("sentinel2", "landsat8", "landsat9", "landsat").

        Returns:
            (12, H, W) Sentinel-2-equivalent surface reflectance.
        """
        if source_sensor == "sentinel2":
            # Already in target format
            if patch.shape[0] == 12:
                return patch[[S2_BAND_ORDER.index(band) for band in self.target_bands]]
            else:
                raise ValueError(
                    f"Sentinel-2 data should have 12 bands, got {patch.shape[0]}"
                )

        elif source_sensor in ("landsat", "landsat8", "landsat9"):
            return self.harmonize_landsat(patch)

        else:
            raise ValueError(f"Unsupported source sensor: {source_sensor}")

    def harmonize_time_series(
        self,
        patches: list[np.ndarray],
        sensors: list[str],
    ) -> np.ndarray:
        """Harmonize a mixed-sensor time series to uniform 12-band S2-equivalent.

        Args:
            patches: List of (C, H, W) patches from different sensors.
            sensors: List of sensor identifiers, one per patch.

        Returns:
            (T, 12, H, W) harmonized time series.
        """
        if len(patches) != len(sensors):
            raise ValueError("patches and sensors must have the same length")

        harmonized = []
        for patch, sensor in zip(patches, sensors):
            h = self.harmonize(patch, sensor)
            harmonized.append(h)

        return np.stack(harmonized, axis=0)


# ─── Convenience Functions ────────────────────────────────────────────────────

def landsat_to_sentinel2(patch: np.ndarray) -> np.ndarray:
    """Quick conversion: 7-band Landsat → 12-band S2-equivalent.

    Args:
        patch: (7, H, W) or (T, 7, H, W) Landsat surface reflectance.

    Returns:
        (12, H, W) or (T, 12, H, W) Sentinel-2-equivalent reflectance.
    """
    harmonizer = BandHarmonizer()

    if patch.ndim == 3:
        return harmonizer.harmonize_landsat(patch)
    elif patch.ndim == 4:
        return np.stack(
            [harmonizer.harmonize_landsat(patch[t]) for t in range(patch.shape[0])],
            axis=0,
        )
    else:
        raise ValueError(f"Expected 3D or 4D input, got {patch.ndim}D")

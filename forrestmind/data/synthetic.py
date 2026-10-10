"""
Synthetic satellite data generator for ForestMind.

Generates physically-plausible multi-temporal, multi-spectral satellite patches
simulating forest degradation dynamics. The generator produces:
  - 12-band optical data (Sentinel-2 equivalent)
  - 2-channel SAR data (VV/VH polarization)
  - Degradation trajectories with seasonal modulation
  - Driver-specific spatial signatures
  - Labels for health state, driver attribution, and trajectory

The synthetic data doesn't need to be photorealistic — it needs to have the
correct *statistical structure*: spectral correlations, temporal dynamics,
and spatial patterns that match real degradation processes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ─── Spectral Profiles ───────────────────────────────────────────────────────
# Approximate reflectance values for Sentinel-2 bands (12 bands)
# Bands: B1(443), B2(490), B3(560), B4(665), B5(705), B6(740),
#        B7(783), B8(842), B8A(865), B9(945), B11(1610), B12(2190)
SPECTRAL_PROFILES = {
    "dense_forest": np.array(
        [0.02, 0.03, 0.05, 0.03, 0.06, 0.25, 0.30, 0.35, 0.33, 0.22, 0.10, 0.05],
        dtype=np.float32,
    ),
    "stressed_forest": np.array(
        [0.03, 0.04, 0.07, 0.05, 0.08, 0.18, 0.22, 0.25, 0.23, 0.18, 0.14, 0.08],
        dtype=np.float32,
    ),
    "degraded_forest": np.array(
        [0.04, 0.06, 0.09, 0.08, 0.10, 0.12, 0.15, 0.18, 0.16, 0.14, 0.18, 0.12],
        dtype=np.float32,
    ),
    "bare_soil": np.array(
        [0.08, 0.10, 0.12, 0.14, 0.15, 0.16, 0.17, 0.18, 0.17, 0.15, 0.22, 0.20],
        dtype=np.float32,
    ),
    "cleared": np.array(
        [0.10, 0.12, 0.14, 0.16, 0.17, 0.17, 0.16, 0.15, 0.14, 0.13, 0.25, 0.22],
        dtype=np.float32,
    ),
    "water": np.array(
        [0.05, 0.06, 0.05, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.00, 0.00],
        dtype=np.float32,
    ),
    "agriculture": np.array(
        [0.04, 0.05, 0.08, 0.06, 0.12, 0.22, 0.26, 0.28, 0.26, 0.20, 0.15, 0.09],
        dtype=np.float32,
    ),
}

# SAR backscatter profiles (VV, VH) in dB, normalized to [0, 1]
SAR_PROFILES = {
    "dense_forest": np.array([0.65, 0.50], dtype=np.float32),
    "stressed_forest": np.array([0.55, 0.40], dtype=np.float32),
    "degraded_forest": np.array([0.45, 0.30], dtype=np.float32),
    "bare_soil": np.array([0.30, 0.15], dtype=np.float32),
    "cleared": np.array([0.25, 0.12], dtype=np.float32),
    "water": np.array([0.05, 0.03], dtype=np.float32),
    "agriculture": np.array([0.40, 0.28], dtype=np.float32),
}

# ─── Health State Mapping ─────────────────────────────────────────────────────
HEALTH_STATES = ["healthy", "stressed", "degraded", "severely_degraded", "cleared"]

# ─── Driver Classes ───────────────────────────────────────────────────────────
DRIVER_CLASSES = [
    "agriculture",
    "cattle_pasture",
    "mining",
    "logging",
    "road_construction",
    "wildfire",
    "drought",
    "urban_expansion",
]


@dataclass
class DegradationScenario:
    """Defines a degradation trajectory for a synthetic patch."""

    initial_health: float  # 0.0 (cleared) to 1.0 (pristine)
    final_health: float
    drivers: list[int]  # indices into DRIVER_CLASSES
    driver_strengths: list[float]  # [0, 1] per driver
    transition_start: float  # fraction of time series where degradation begins
    transition_speed: float  # steepness of sigmoid transition
    seasonal_amplitude: float  # seasonal NDVI variation
    drought_events: list[tuple[int, float]]  # (timestep, severity)


def _perlin_noise_2d(shape: tuple[int, int], scale: float = 32.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate 2D Perlin-like noise via interpolated random gradients.
    
    This is a simplified fractal noise generator. For the prototype,
    it provides spatially coherent noise patterns for canopy structure.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    h, w = shape
    # Grid of random gradients at coarse resolution
    grid_h = max(2, int(h / scale) + 2)
    grid_w = max(2, int(w / scale) + 2)
    angles = rng.uniform(0, 2 * math.pi, (grid_h, grid_w)).astype(np.float32)
    gradients = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    # Pixel coordinates in gradient space
    y_coords = np.linspace(0, grid_h - 1.001, h, dtype=np.float32)
    x_coords = np.linspace(0, grid_w - 1.001, w, dtype=np.float32)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")

    # Integer grid cell and fractional position
    y0 = yy.astype(np.int32)
    x0 = xx.astype(np.int32)
    y1 = y0 + 1
    x1 = x0 + 1
    fy = yy - y0
    fx = xx - x0

    # Smoothstep interpolation
    fy = fy * fy * (3 - 2 * fy)
    fx = fx * fx * (3 - 2 * fx)

    # Dot products with gradients at 4 corners
    def dot_grad(gy, gx, dy, dx):
        g = gradients[gy, gx]
        return g[..., 0] * dy + g[..., 1] * dx

    n00 = dot_grad(y0, x0, fy, fx)
    n10 = dot_grad(y1, x0, fy - 1, fx)
    n01 = dot_grad(y0, x1, fy, fx - 1)
    n11 = dot_grad(y1, x1, fy - 1, fx - 1)

    # Bilinear interpolation
    nx0 = n00 * (1 - fx) + n01 * fx
    nx1 = n10 * (1 - fx) + n11 * fx
    noise = nx0 * (1 - fy) + nx1 * fy

    # Normalize to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise


def _fractal_noise_2d(
    shape: tuple[int, int],
    octaves: int = 4,
    persistence: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Multi-octave fractal noise for realistic canopy texture."""
    if rng is None:
        rng = np.random.default_rng()
    
    noise = np.zeros(shape, dtype=np.float32)
    amplitude = 1.0
    scale = 32.0
    total_amplitude = 0.0

    for _ in range(octaves):
        noise += amplitude * _perlin_noise_2d(shape, scale=scale, rng=rng)
        total_amplitude += amplitude
        amplitude *= persistence
        scale /= 2.0

    return noise / total_amplitude


def _generate_driver_mask(
    shape: tuple[int, int],
    driver: str,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a spatial mask for a specific degradation driver.
    
    Different drivers create characteristic spatial patterns:
    - agriculture: rectangular blocks
    - cattle_pasture: large irregular patches
    - mining: circular pits
    - logging: dendritic patterns following ridges
    - road_construction: linear corridors
    - wildfire: irregular spreading patterns
    - drought: diffuse, affects entire patch
    - urban_expansion: radial from edges
    """
    h, w = shape
    mask = np.zeros(shape, dtype=np.float32)

    if driver == "agriculture":
        # Rectangular blocks
        num_blocks = rng.integers(1, 5)
        for _ in range(num_blocks):
            bh = rng.integers(h // 6, h // 3)
            bw = rng.integers(w // 6, w // 3)
            by = rng.integers(0, h - bh)
            bx = rng.integers(0, w - bw)
            mask[by : by + bh, bx : bx + bw] = 1.0

    elif driver == "cattle_pasture":
        # Large irregular patches
        base_noise = _fractal_noise_2d(shape, octaves=3, rng=rng)
        threshold = np.percentile(base_noise, 100 * (1 - strength * 0.5))
        mask = (base_noise > threshold).astype(np.float32)

    elif driver == "mining":
        # Circular pits
        num_pits = rng.integers(1, 4)
        yy, xx = np.mgrid[0:h, 0:w]
        for _ in range(num_pits):
            cy, cx = rng.integers(h // 4, 3 * h // 4), rng.integers(w // 4, 3 * w // 4)
            radius = rng.integers(5, max(6, h // 8))
            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            mask += np.clip(1.0 - dist / radius, 0, 1)
        mask = np.clip(mask, 0, 1)

    elif driver == "logging":
        # Dendritic / branching patterns following noise ridges
        noise = _fractal_noise_2d(shape, octaves=5, rng=rng)
        # Extract ridge lines
        grad_y = np.gradient(noise, axis=0)
        grad_x = np.gradient(noise, axis=1)
        ridge = 1.0 / (1.0 + 10.0 * (grad_y ** 2 + grad_x ** 2))
        mask = (ridge > np.percentile(ridge, 85)).astype(np.float32)

    elif driver == "road_construction":
        # Linear corridors
        num_roads = rng.integers(1, 4)
        yy, xx = np.mgrid[0:h, 0:w]
        for _ in range(num_roads):
            # Random line parameterized by angle and offset
            angle = rng.uniform(0, math.pi)
            offset = rng.uniform(-0.3, 0.3)
            dist = np.abs(
                np.cos(angle) * (xx / w - 0.5) + np.sin(angle) * (yy / h - 0.5) - offset
            )
            road_width = rng.uniform(0.01, 0.03)
            mask += np.clip(1.0 - dist / road_width, 0, 1)
        mask = np.clip(mask, 0, 1)

    elif driver == "wildfire":
        # Irregular spreading from ignition points
        noise = _fractal_noise_2d(shape, octaves=4, rng=rng)
        num_ignitions = rng.integers(1, 3)
        yy, xx = np.mgrid[0:h, 0:w]
        fire_mask = np.zeros(shape, dtype=np.float32)
        for _ in range(num_ignitions):
            cy, cx = rng.integers(0, h), rng.integers(0, w)
            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            spread_radius = rng.uniform(h * 0.2, h * 0.5)
            spread = np.clip(1.0 - dist / spread_radius, 0, 1)
            # Modulate by noise for irregular edges
            fire_mask += spread * (0.3 + 0.7 * noise)
        mask = np.clip(fire_mask, 0, 1)

    elif driver == "drought":
        # Diffuse, affects the whole patch with spatial variation
        noise = _fractal_noise_2d(shape, octaves=2, rng=rng)
        mask = 0.5 + 0.5 * noise  # 0.5-1.0 range: drought is spatially uniform-ish

    elif driver == "urban_expansion":
        # Radial expansion from patch edges
        yy, xx = np.mgrid[0:h, 0:w]
        dist_to_edge = np.minimum(
            np.minimum(yy, h - 1 - yy),
            np.minimum(xx, w - 1 - xx),
        ).astype(np.float32)
        max_dist = min(h, w) / 2
        mask = np.clip(1.0 - dist_to_edge / (max_dist * 0.3), 0, 1)
        # Add some noise for realism
        noise = _fractal_noise_2d(shape, octaves=3, rng=rng)
        mask *= (0.5 + 0.5 * noise)
        mask = np.clip(mask, 0, 1)

    return mask * strength


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


class SyntheticForestGenerator:
    """Generate synthetic multi-temporal satellite data for ForestMind.
    
    Each sample contains:
      - optical: (T, 12, H, W) multi-spectral imagery
      - sar: (T, 2, H, W) SAR backscatter
      - timestamps: (T,) relative day offsets
      - health_state: int (0-4) final health classification
      - health_score: float (0-100) continuous health score
      - drivers: (8,) binary driver presence
      - driver_strengths: (8,) continuous driver intensity
      - trajectory: (5,) future health scores at prediction horizons
    """

    def __init__(
        self,
        patch_size: int = 128,
        num_bands: int = 12,
        sar_channels: int = 2,
        time_steps: int = 12,
        seed: int = 42,
    ):
        self.patch_size = patch_size
        self.num_bands = num_bands
        self.sar_channels = sar_channels
        self.time_steps = time_steps
        self.rng = np.random.default_rng(seed)

    def _create_scenario(self) -> DegradationScenario:
        """Create a random but plausible degradation scenario."""
        # Randomly choose the degradation severity
        scenario_type = self.rng.choice(
            ["stable_healthy", "mild_stress", "moderate_degradation",
             "severe_degradation", "clearance"],
            p=[0.25, 0.20, 0.25, 0.15, 0.15],
        )

        if scenario_type == "stable_healthy":
            initial = self.rng.uniform(0.80, 1.00)
            final = initial - self.rng.uniform(0.0, 0.05)
            drivers = []
            driver_strengths = []
        elif scenario_type == "mild_stress":
            initial = self.rng.uniform(0.70, 0.95)
            final = initial - self.rng.uniform(0.05, 0.20)
            num_drivers = self.rng.integers(1, 3)
            drivers = self.rng.choice(len(DRIVER_CLASSES), size=num_drivers, replace=False).tolist()
            driver_strengths = self.rng.uniform(0.1, 0.4, size=num_drivers).tolist()
        elif scenario_type == "moderate_degradation":
            initial = self.rng.uniform(0.60, 0.90)
            final = initial - self.rng.uniform(0.20, 0.40)
            num_drivers = self.rng.integers(1, 4)
            drivers = self.rng.choice(len(DRIVER_CLASSES), size=num_drivers, replace=False).tolist()
            driver_strengths = self.rng.uniform(0.3, 0.7, size=num_drivers).tolist()
        elif scenario_type == "severe_degradation":
            initial = self.rng.uniform(0.50, 0.85)
            final = initial - self.rng.uniform(0.35, 0.55)
            final = max(final, 0.05)
            num_drivers = self.rng.integers(2, 5)
            drivers = self.rng.choice(len(DRIVER_CLASSES), size=num_drivers, replace=False).tolist()
            driver_strengths = self.rng.uniform(0.5, 0.9, size=num_drivers).tolist()
        else:  # clearance
            initial = self.rng.uniform(0.60, 0.95)
            final = self.rng.uniform(0.0, 0.15)
            num_drivers = self.rng.integers(1, 3)
            # Clearance drivers are typically agriculture, cattle, mining, or urban
            clear_drivers = [0, 1, 2, 7]
            drivers = self.rng.choice(clear_drivers, size=min(num_drivers, len(clear_drivers)), replace=False).tolist()
            driver_strengths = self.rng.uniform(0.7, 1.0, size=len(drivers)).tolist()

        # Drought events (random acute stress)
        num_droughts = self.rng.integers(0, 3)
        drought_events = [
            (int(self.rng.integers(0, self.time_steps)), float(self.rng.uniform(0.05, 0.20)))
            for _ in range(num_droughts)
        ]

        return DegradationScenario(
            initial_health=float(initial),
            final_health=float(final),
            drivers=drivers,
            driver_strengths=driver_strengths,
            transition_start=float(self.rng.uniform(0.1, 0.5)),
            transition_speed=float(self.rng.uniform(3.0, 12.0)),
            seasonal_amplitude=float(self.rng.uniform(0.02, 0.08)),
            drought_events=drought_events,
        )

    def _health_trajectory(self, scenario: DegradationScenario) -> np.ndarray:
        """Compute health score trajectory over time steps."""
        t_norm = np.linspace(0, 1, self.time_steps, dtype=np.float32)

        # Sigmoid degradation curve
        degradation_progress = _sigmoid(
            scenario.transition_speed * (t_norm - scenario.transition_start)
        )
        # Normalize so it goes from 0 at start to 1 at end
        degradation_progress = (degradation_progress - degradation_progress[0]) / (
            degradation_progress[-1] - degradation_progress[0] + 1e-8
        )

        # Interpolate health
        health = (
            scenario.initial_health
            + (scenario.final_health - scenario.initial_health) * degradation_progress
        )

        # Add seasonal modulation (sinusoidal with period = time_steps)
        seasonal = scenario.seasonal_amplitude * np.sin(
            2 * math.pi * t_norm * (1 + self.rng.uniform(-0.2, 0.2))
        )
        health = health + seasonal

        # Apply drought events as temporary dips
        for timestep, severity in scenario.drought_events:
            if 0 <= timestep < self.time_steps:
                # Gaussian dip centered on the drought timestep
                t_idx = np.arange(self.time_steps)
                dip = severity * np.exp(-0.5 * ((t_idx - timestep) / 1.5) ** 2)
                health = health - dip

        return np.clip(health, 0.0, 1.0).astype(np.float32)

    def _health_to_spectral(
        self,
        health: float,
        driver_masks: np.ndarray,
        base_canopy: np.ndarray,
    ) -> np.ndarray:
        """Convert health values to multi-spectral reflectance.
        
        Args:
            health: scalar health value for this timestep
            driver_masks: (H, W) combined spatial degradation mask
            base_canopy: (H, W) base canopy density
            
        Returns:
            (12, H, W) spectral reflectance
        """
        h, w = base_canopy.shape

        # Pixel-level health modulated by spatial patterns
        pixel_health = health * base_canopy * (1.0 - driver_masks * (1.0 - health))
        pixel_health = np.clip(pixel_health, 0.0, 1.0)

        # Blend spectral profiles based on pixel health
        spectral = np.zeros((self.num_bands, h, w), dtype=np.float32)
        
        # Define health thresholds for profile blending
        profiles = [
            (0.8, 1.0, SPECTRAL_PROFILES["dense_forest"]),
            (0.6, 0.8, SPECTRAL_PROFILES["stressed_forest"]),
            (0.4, 0.6, SPECTRAL_PROFILES["degraded_forest"]),
            (0.15, 0.4, SPECTRAL_PROFILES["bare_soil"]),
            (0.0, 0.15, SPECTRAL_PROFILES["cleared"]),
        ]

        for low, high, profile in profiles:
            mask = ((pixel_health >= low) & (pixel_health < high)).astype(np.float32)
            # Smooth blending within range
            blend = (pixel_health - low) / (high - low + 1e-8)
            blend = np.clip(blend, 0, 1) * mask
            spectral += profile[:, None, None] * blend[None, :, :]

        # Handle pixels exactly at 1.0
        mask_top = (pixel_health >= 1.0).astype(np.float32)
        spectral += SPECTRAL_PROFILES["dense_forest"][:, None, None] * mask_top[None, :, :]

        # Add band-correlated noise
        noise = self.rng.normal(0, 0.005, spectral.shape).astype(np.float32)
        spectral = np.clip(spectral + noise, 0.0, 1.0)

        return spectral

    def _health_to_sar(
        self,
        health: float,
        driver_masks: np.ndarray,
        base_canopy: np.ndarray,
    ) -> np.ndarray:
        """Convert health values to SAR backscatter."""
        h, w = base_canopy.shape
        pixel_health = health * base_canopy * (1.0 - driver_masks * (1.0 - health))
        pixel_health = np.clip(pixel_health, 0.0, 1.0)

        sar = np.zeros((self.sar_channels, h, w), dtype=np.float32)

        # Blend SAR profiles
        profiles = [
            (0.8, 1.0, SAR_PROFILES["dense_forest"]),
            (0.6, 0.8, SAR_PROFILES["stressed_forest"]),
            (0.4, 0.6, SAR_PROFILES["degraded_forest"]),
            (0.15, 0.4, SAR_PROFILES["bare_soil"]),
            (0.0, 0.15, SAR_PROFILES["cleared"]),
        ]

        for low, high, profile in profiles:
            mask = ((pixel_health >= low) & (pixel_health < high)).astype(np.float32)
            blend = (pixel_health - low) / (high - low + 1e-8)
            blend = np.clip(blend, 0, 1) * mask
            sar += profile[:, None, None] * blend[None, :, :]

        mask_top = (pixel_health >= 1.0).astype(np.float32)
        sar += SAR_PROFILES["dense_forest"][:, None, None] * mask_top[None, :, :]

        # SAR speckle noise
        speckle = self.rng.exponential(1.0, sar.shape).astype(np.float32)
        sar = sar * (0.7 + 0.3 * speckle)
        return np.clip(sar, 0.0, 1.0)

    def _compute_future_trajectory(
        self,
        scenario: DegradationScenario,
        health_trajectory: np.ndarray,
        horizons_days: list[int] = [30, 180, 365, 1095, 1825],
    ) -> np.ndarray:
        """Extrapolate health trajectory into the future.
        
        Uses the degradation trend from the observed trajectory to predict
        future health scores at specified horizons.
        """
        # Estimate current degradation rate from last few timesteps
        if len(health_trajectory) >= 3:
            recent_rate = (health_trajectory[-1] - health_trajectory[-3]) / 2.0
        else:
            recent_rate = health_trajectory[-1] - health_trajectory[0]

        # Days per timestep (assume monthly = 30 days)
        days_per_step = 30.0
        current_health = health_trajectory[-1]

        future_health = []
        for horizon in horizons_days:
            steps_ahead = horizon / days_per_step
            # Non-linear extrapolation: degradation accelerates as ecosystem weakens
            # (this models the tipping-point dynamics)
            acceleration = 1.0 + 0.5 * max(0, 0.5 - current_health)
            predicted = current_health + recent_rate * steps_ahead * acceleration
            # Add some noise to future predictions
            noise = self.rng.normal(0, 0.02 * (horizon / 365.0))
            predicted = np.clip(predicted + noise, 0.0, 1.0)
            future_health.append(float(predicted))
            # Update for next horizon
            current_health = predicted

        return np.array(future_health, dtype=np.float32)

    def _classify_health(self, health_score: float) -> int:
        """Convert continuous health score to discrete state.
        
        Returns:
            0=healthy, 1=stressed, 2=degraded, 3=severely_degraded, 4=cleared
        """
        if health_score >= 0.80:
            return 0
        elif health_score >= 0.60:
            return 1
        elif health_score >= 0.40:
            return 2
        elif health_score >= 0.15:
            return 3
        else:
            return 4

    def generate_sample(self) -> dict[str, np.ndarray | int | float]:
        """Generate a single synthetic multi-temporal satellite sample.
        
        Returns:
            Dictionary containing:
              optical: (T, 12, H, W) multi-spectral reflectance
              sar: (T, 2, H, W) SAR backscatter
              timestamps: (T,) day offsets from start
              health_state: int (0-4)
              health_score: float (0-100)
              drivers: (8,) binary driver presence
              driver_strengths: (8,) continuous [0, 1]
              trajectory: (5,) future health scores
              health_history: (T,) health score at each timestep
        """
        shape = (self.patch_size, self.patch_size)
        scenario = self._create_scenario()

        # Generate base canopy structure (persistent across time)
        base_canopy = _fractal_noise_2d(shape, octaves=4, rng=self.rng)
        # Threshold to create canopy vs gaps
        canopy_density = float(self.rng.uniform(0.7, 0.98))
        base_canopy = np.clip(base_canopy * canopy_density + (1 - canopy_density), 0, 1)

        # Generate driver-specific spatial masks
        combined_driver_mask = np.zeros(shape, dtype=np.float32)
        for driver_idx, strength in zip(scenario.drivers, scenario.driver_strengths):
            driver_name = DRIVER_CLASSES[driver_idx]
            mask = _generate_driver_mask(shape, driver_name, strength, self.rng)
            combined_driver_mask = np.maximum(combined_driver_mask, mask)
        combined_driver_mask = np.clip(combined_driver_mask, 0.0, 1.0)

        # Compute temporal health trajectory
        health_trajectory = self._health_trajectory(scenario)

        # Generate multi-temporal data
        optical_series = np.zeros(
            (self.time_steps, self.num_bands, *shape), dtype=np.float32
        )
        sar_series = np.zeros(
            (self.time_steps, self.sar_channels, *shape), dtype=np.float32
        )

        for t in range(self.time_steps):
            h_t = health_trajectory[t]
            # Temporal evolution of driver mask (grows over time)
            temporal_mask_scale = min(1.0, t / max(1, self.time_steps - 1))
            current_driver_mask = combined_driver_mask * temporal_mask_scale

            optical_series[t] = self._health_to_spectral(h_t, current_driver_mask, base_canopy)
            sar_series[t] = self._health_to_sar(h_t, current_driver_mask, base_canopy)

        # Timestamps as day offsets (monthly spacing with jitter)
        timestamps = np.array(
            [t * 30 + int(self.rng.integers(-5, 6)) for t in range(self.time_steps)],
            dtype=np.float32,
        )
        timestamps = np.sort(timestamps)
        timestamps -= timestamps[0]  # start from 0

        # Labels
        final_health = health_trajectory[-1]
        health_state = self._classify_health(final_health)
        health_score = float(final_health * 100.0)

        # Driver labels
        driver_binary = np.zeros(len(DRIVER_CLASSES), dtype=np.float32)
        driver_continuous = np.zeros(len(DRIVER_CLASSES), dtype=np.float32)
        for idx, strength in zip(scenario.drivers, scenario.driver_strengths):
            driver_binary[idx] = 1.0
            driver_continuous[idx] = strength

        # Future trajectory
        trajectory = self._compute_future_trajectory(
            scenario, health_trajectory
        )

        return {
            "optical": optical_series,
            "sar": sar_series,
            "timestamps": timestamps,
            "health_state": health_state,
            "health_score": health_score,
            "drivers": driver_binary,
            "driver_strengths": driver_continuous,
            "trajectory": trajectory,
            "health_history": health_trajectory,
        }

    def generate_dataset(
        self,
        num_samples: int,
        output_dir: Optional[str | Path] = None,
    ) -> list[dict]:
        """Generate a full dataset of synthetic samples.
        
        Args:
            num_samples: Number of samples to generate.
            output_dir: If provided, save samples as .npz files.
            
        Returns:
            List of sample dictionaries.
        """
        samples = []
        for i in range(num_samples):
            sample = self.generate_sample()
            samples.append(sample)

            if output_dir is not None:
                out_path = Path(output_dir)
                out_path.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    out_path / f"sample_{i:06d}.npz",
                    **{k: v if isinstance(v, np.ndarray) else np.array(v) for k, v in sample.items()},
                )

        return samples

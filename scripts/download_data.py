#!/usr/bin/env python3
"""
Download real satellite data for ForestMind.

CLI tool to fetch, preprocess, and cache satellite imagery from
Microsoft Planetary Computer for a specified study region and time range.

Usage:
  # Download Sentinel-2 for a region defined in config
  python scripts/download_data.py --config configs/sentinel2.yaml

  # Download specific sensors for custom region
  python scripts/download_data.py \
    --sensors s2,s1 \
    --region -63.5,-11.5,-62.5,-10.5 \
    --start 2023-01-01 \
    --end 2024-01-01 \
    --cache-dir data/real

  # Download Landsat historical data
  python scripts/download_data.py --config configs/landsat.yaml --sensors landsat

Requires:
  pip install forestmind[real-data]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forestmind.utils.geo import BoundingBox

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_data")

# Sensor name aliases
SENSOR_ALIASES = {
    "s2": "sentinel2",
    "sentinel2": "sentinel2",
    "sentinel-2": "sentinel2",
    "s1": "sentinel1",
    "sentinel1": "sentinel1",
    "sentinel-1": "sentinel1",
    "landsat": "landsat",
    "ls": "landsat",
    "landsat8": "landsat",
    "landsat9": "landsat",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download satellite data for ForestMind",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Config YAML file with region and sensor settings",
    )
    parser.add_argument(
        "--sensors", type=str, default="s2,s1",
        help="Comma-separated sensors to download (s2, s1, landsat)",
    )
    parser.add_argument(
        "--region", type=str, default=None,
        help="Bounding box: west,south,east,north (WGS84 degrees)",
    )
    parser.add_argument(
        "--start", type=str, default="2023-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", type=str, default="2024-01-01",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--max-cloud", type=float, default=25.0,
        help="Maximum cloud cover %% (optical sensors only)",
    )
    parser.add_argument(
        "--max-scenes", type=int, default=100,
        help="Maximum scenes per sensor",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="data/real",
        help="Local cache directory",
    )
    parser.add_argument(
        "--patch-size", type=int, default=128,
        help="Target patch size in pixels",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Search only, don't download",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_region_from_config(config: dict) -> tuple[BoundingBox, str, str]:
    """Extract region, start_date, end_date from config."""
    data_cfg = config.get("data", {})
    region = data_cfg.get("region", {})

    bbox = BoundingBox(
        west=region.get("west", -63.5),
        south=region.get("south", -11.5),
        east=region.get("east", -62.5),
        north=region.get("north", -10.5),
    )
    start = region.get("start_date", "2023-01-01")
    end = region.get("end_date", "2024-01-01")

    return bbox, start, end


def download_sentinel2(
    bbox: BoundingBox,
    start_date: str,
    end_date: str,
    cache_dir: str,
    patch_size: int,
    max_cloud: float,
    max_scenes: int,
    dry_run: bool,
    config: dict | None = None,
) -> int:
    """Download Sentinel-2 data."""
    from forestmind.data.adapters.sentinel2 import Sentinel2Adapter

    s2_cfg = (config or {}).get("data", {}).get("sentinel2", {})
    resolution = s2_cfg.get("resolution", 10)

    adapter = Sentinel2Adapter(
        cache_dir=cache_dir,
        patch_size=patch_size,
        resolution=resolution,
    )

    logger.info(f"🛰️  Searching Sentinel-2 L2A: {start_date} → {end_date}")
    scenes = adapter.search(bbox, start_date, end_date, max_cloud, max_scenes)
    logger.info(f"   Found {len(scenes)} scenes")

    if dry_run:
        for s in scenes[:10]:
            logger.info(f"   • {s.scene_id} | {s.date_str} | cloud: {s.cloud_cover:.1f}%")
        if len(scenes) > 10:
            logger.info(f"   ... and {len(scenes) - 10} more")
        return len(scenes)

    paths = adapter.download(scenes, bbox)
    logger.info(f"   ✓ Downloaded {len(paths)} patches → {cache_dir}/sentinel2/")
    return len(paths)


def download_sentinel1(
    bbox: BoundingBox,
    start_date: str,
    end_date: str,
    cache_dir: str,
    patch_size: int,
    max_scenes: int,
    dry_run: bool,
    config: dict | None = None,
) -> int:
    """Download Sentinel-1 SAR data."""
    from forestmind.data.adapters.sentinel1 import Sentinel1Adapter

    s1_cfg = (config or {}).get("data", {}).get("sentinel1", {})
    orbit = s1_cfg.get("orbit_direction")

    adapter = Sentinel1Adapter(
        cache_dir=cache_dir,
        patch_size=patch_size,
        orbit_direction=orbit,
    )

    logger.info(f"📡 Searching Sentinel-1 GRD: {start_date} → {end_date}")
    scenes = adapter.search(bbox, start_date, end_date, max_results=max_scenes)
    logger.info(f"   Found {len(scenes)} scenes")

    if dry_run:
        for s in scenes[:10]:
            orbit_dir = s.extra.get("orbit_direction", "?")
            logger.info(f"   • {s.scene_id} | {s.date_str} | orbit: {orbit_dir}")
        if len(scenes) > 10:
            logger.info(f"   ... and {len(scenes) - 10} more")
        return len(scenes)

    paths = adapter.download(scenes, bbox)
    logger.info(f"   ✓ Downloaded {len(paths)} patches → {cache_dir}/sentinel1/")
    return len(paths)


def download_landsat(
    bbox: BoundingBox,
    start_date: str,
    end_date: str,
    cache_dir: str,
    patch_size: int,
    max_cloud: float,
    max_scenes: int,
    dry_run: bool,
    config: dict | None = None,
) -> int:
    """Download Landsat data."""
    from forestmind.data.adapters.landsat import LandsatAdapter

    ls_cfg = (config or {}).get("data", {}).get("landsat", {})
    resolution = ls_cfg.get("resolution", 30)
    platforms = ls_cfg.get("platforms", ["landsat-8", "landsat-9"])

    adapter = LandsatAdapter(
        cache_dir=cache_dir,
        patch_size=patch_size,
        resolution=resolution,
        platforms=platforms,
    )

    logger.info(f"🌎 Searching Landsat: {start_date} → {end_date}")
    scenes = adapter.search(bbox, start_date, end_date, max_cloud, max_scenes)
    logger.info(f"   Found {len(scenes)} scenes")

    if dry_run:
        for s in scenes[:10]:
            platform = s.extra.get("platform", "?")
            logger.info(f"   • {s.scene_id} | {s.date_str} | {platform} | cloud: {s.cloud_cover:.1f}%")
        if len(scenes) > 10:
            logger.info(f"   ... and {len(scenes) - 10} more")
        return len(scenes)

    paths = adapter.download(scenes, bbox)
    logger.info(f"   ✓ Downloaded {len(paths)} patches → {cache_dir}/landsat/")
    return len(paths)


def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args()

    # Load config if provided
    config = None
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded config: {args.config}")

    # Determine region
    if args.region:
        parts = [float(x) for x in args.region.split(",")]
        if len(parts) != 4:
            logger.error("Region must be: west,south,east,north")
            sys.exit(1)
        bbox = BoundingBox(west=parts[0], south=parts[1], east=parts[2], north=parts[3])
        start_date = args.start
        end_date = args.end
    elif config:
        bbox, start_date, end_date = get_region_from_config(config)
    else:
        logger.error("Specify --region or --config")
        sys.exit(1)

    logger.info(f"Region: ({bbox.west:.2f}, {bbox.south:.2f}) → ({bbox.east:.2f}, {bbox.north:.2f})")
    logger.info(f"  Area: ~{bbox.approx_area_km2():.0f} km²")
    logger.info(f"  Time: {start_date} → {end_date}")
    logger.info(f"  Cache: {args.cache_dir}")

    if args.dry_run:
        logger.info("  🔍 DRY RUN — search only, no downloads")

    # Parse sensors
    sensor_list = [SENSOR_ALIASES.get(s.strip().lower(), s.strip().lower())
                   for s in args.sensors.split(",")]

    start_time = time.time()
    total_downloaded = 0

    # Download each sensor
    for sensor in sensor_list:
        try:
            if sensor == "sentinel2":
                n = download_sentinel2(
                    bbox, start_date, end_date, args.cache_dir, args.patch_size,
                    args.max_cloud, args.max_scenes, args.dry_run, config,
                )
            elif sensor == "sentinel1":
                n = download_sentinel1(
                    bbox, start_date, end_date, args.cache_dir, args.patch_size,
                    args.max_scenes, args.dry_run, config,
                )
            elif sensor == "landsat":
                n = download_landsat(
                    bbox, start_date, end_date, args.cache_dir, args.patch_size,
                    args.max_cloud, args.max_scenes, args.dry_run, config,
                )
            else:
                logger.warning(f"Unknown sensor: {sensor}")
                continue

            total_downloaded += n

        except ImportError as e:
            logger.error(f"Missing dependencies for {sensor}: {e}")
            logger.error("Install with: pip install forestmind[real-data]")
        except Exception as e:
            logger.error(f"Failed to download {sensor}: {e}")

    elapsed = time.time() - start_time
    action = "found" if args.dry_run else "downloaded"
    logger.info(f"\n{'='*50}")
    logger.info(f"  Total {action}: {total_downloaded} patches")
    logger.info(f"  Time: {elapsed:.1f}s")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()

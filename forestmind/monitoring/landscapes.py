"""Reproducible worldwide study rectangles, paged into bounded monitoring tiles.

A plan is a geometry index, not a forest inventory or a request to download data.
Only its parameters are persisted. Even a world-sized plan materializes at most
one requested page of tiles; row descriptors scale with latitude, not land area.
"""
from __future__ import annotations

from bisect import bisect_right
from datetime import datetime, timezone
import hashlib
import json
import math


METHOD = "spherical-rectangle-rows-v1"
RADIUS_KM = 6371.0088
MAX_PLANS = 100
MAX_PAGE = 100
MIN_AREA_KM2 = .0001
NOTE = ("Tiles partition the selected study rectangle, not verified forest extent. "
        "Saved non-synthetic observations indicate data presence only, not complete "
        "pixel or temporal coverage, healthy forest, or validated predictions. "
        "Imported observations retain the user's source declaration.")
MEASUREMENTS = ("ndvi", "ndmi", "nbr", "precipitation_anomaly", "temperature_anomaly",
                "sar_change", "forest_cover", "fire_anomaly")


def rectangle_area(bbox):
    w, s, e, n = bbox
    return RADIUS_KM ** 2 * math.radians(e - w) * (math.sin(math.radians(n)) - math.sin(math.radians(s)))


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validated(values):
    name = values.get("name", "")
    country = values.get("country", "Unspecified")
    if not isinstance(name, str) or not 2 <= len(name.strip()) <= 120:
        raise ValueError("Name must contain 2–120 characters")
    if not isinstance(country, str) or not 2 <= len(country.strip()) <= 100:
        raise ValueError("Country / location must contain 2–100 characters")
    bbox = values.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or not all(_number(v) for v in bbox):
        raise ValueError("Enter four finite WGS84 coordinates: west, south, east, north")
    w, s, e, n = map(float, bbox)
    if not (-180 <= w <= 180 and -180 <= e <= 180 and -85 <= s < n <= 85):
        raise ValueError("Longitude must be −180 to 180 and south < north within 85°S–85°N")
    if w == e or (w == 180 and e == -180):
        raise ValueError("West and east must enclose a nonzero span; use −180 to 180 for all longitudes")
    target = values.get("tile_area_km2", 25)
    if not _number(target) or not 1 <= target <= 25:
        raise ValueError("Tile area must be between 1 and 25 km²")
    return {"name": name.strip(), "country": country.strip(), "bbox": [w, s, e, n],
            "tile_area_km2": float(target)}


class TileGrid:
    """Deterministic south-to-north rows; each row follows the requested longitude span."""

    def __init__(self, bbox, target):
        w, south, e, north = bbox
        segments = [(w, e)] if w < e else [(w, 180.), (-180., e)]
        segments = [(a, b) for a, b in segments if b > a]
        # A numerical margin keeps emitted tiles below the existing strict
        # acquisition area bound despite subtraction at large coordinates.
        effective_target = target * (1 - 1e-9)
        rows = max(1, math.ceil((north - south) / math.degrees(math.sqrt(effective_target) / RADIUS_KM)))
        self.rows = []
        self.starts = []
        self.south_edges = []
        self.tile_count = 0
        self.area_km2 = sum(rectangle_area([a, south, b, north]) for a, b in segments)
        for row in range(rows):
            s = south + (north - south) * row / rows
            n = north if row == rows - 1 else south + (north - south) * (row + 1) / rows
            parts = []
            self.starts.append(self.tile_count)
            self.south_edges.append(s)
            for a, b in segments:
                area = rectangle_area([a, s, b, n])
                columns = max(1, math.ceil(area / effective_target), math.ceil(b - a))
                # Extremely thin pieces cannot satisfy the importer's minimum
                # polygon/area contract. Reject the plan, never omit its edge.
                if area / columns < MIN_AREA_KM2 or (b - a) * (n - s) / columns < 2e-8:
                    raise ValueError("The rectangle has an edge too narrow for valid monitoring tiles; widen or split its boundary")
                parts.append((a, b, columns, self.tile_count))
                self.tile_count += columns
            self.rows.append((s, n, parts))

    def tile(self, index):
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < self.tile_count:
            raise KeyError("Landscape tile not found")
        row = bisect_right(self.starts, index) - 1
        s, n, parts = self.rows[row]
        for a, b, columns, start in parts:
            if start <= index < start + columns:
                column = index - start
                w = a + (b - a) * column / columns
                e = b if column == columns - 1 else a + (b - a) * (column + 1) / columns
                bbox = [w, s, e, n]
                return {"index": index, "bbox": bbox, "area_km2": rectangle_area(bbox)}
        raise KeyError("Landscape tile not found")  # Defensive against damaged grid state.

    def index_for(self, bbox):
        """Locate an exact saved tile without enumerating millions of planned cells."""
        w, s, e, n = bbox
        row = bisect_right(self.south_edges, s) - 1
        if row < 0 or self.rows[row][0] != s or self.rows[row][1] != n:
            return None
        for a, b, columns, start in self.rows[row][2]:
            if a <= w < e <= b:
                column = round((w - a) * columns / (b - a))
                if 0 <= column < columns:
                    index = start + column
                    if self.tile(index)["bbox"] == bbox:
                        return index
        return None


def _rectangle_bbox(geometry):
    """Accept exactly one rectangular footprint, including rotated/reversed rings."""
    if not isinstance(geometry, dict) or geometry.get("type") != "Polygon":
        return None
    rings = geometry.get("coordinates")
    if not isinstance(rings, list) or len(rings) != 1:
        return None
    ring = rings[0]
    if not isinstance(ring, list) or len(ring) != 5 or ring[0] != ring[-1]:
        return None
    if any(not isinstance(p, list) or len(p) != 2 or not all(_number(v) for v in p) for p in ring):
        return None
    w, e = min(p[0] for p in ring), max(p[0] for p in ring)
    s, n = min(p[1] for p in ring), max(p[1] for p in ring)
    corners = {(w, s), (w, n), (e, s), (e, n)}
    if len(corners) != 4 or {tuple(p) for p in ring[:-1]} != corners:
        return None
    if any(a[0] != b[0] and a[1] != b[1] for a, b in zip(ring, ring[1:])):
        return None
    return [w, s, e, n]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class LandscapeService:
    def __init__(self, store):
        self.store = store
        with store.connection() as db:
            db.execute("CREATE TABLE IF NOT EXISTS landscape_plans (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")

    def create(self, values):
        params = _validated(values)
        grid = TileGrid(params["bbox"], params["tile_area_km2"])
        plan = {**params, "method": METHOD, "earth_radius_km": RADIUS_KM,
                "tile_order": "south_to_north_rows_then_west_to_east_with_dateline_split",
                "area_km2": grid.area_km2, "tile_count": grid.tile_count,
                "row_count": len(grid.rows), "crosses_antimeridian": params["bbox"][0] > params["bbox"][2],
                "verified_forest_boundary": False, "note": NOTE}
        digest = _digest(plan)
        plan.update(id="landscape-" + digest[:24], plan_sha256=digest,
                    created_at=datetime.now(timezone.utc).isoformat())
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            saved = db.execute("SELECT payload FROM landscape_plans WHERE id=?", (plan["id"],)).fetchone()
            if saved:
                return json.loads(saved[0])
            if db.execute("SELECT COUNT(*) FROM landscape_plans").fetchone()[0] >= MAX_PLANS:
                raise ValueError(f"This local workspace supports at most {MAX_PLANS} saved landscape plans")
            db.execute("INSERT INTO landscape_plans VALUES (?,?)", (plan["id"], self.store._json(plan)))
        return plan

    def list(self):
        with self.store.connection() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT payload FROM landscape_plans ORDER BY rowid DESC LIMIT ?", (MAX_PLANS,))]

    def get(self, landscape_id):
        with self.store.connection() as db:
            row = db.execute("SELECT payload FROM landscape_plans WHERE id=?", (landscape_id,)).fetchone()
        if row is None:
            raise KeyError("Landscape plan not found")
        plan = json.loads(row[0])
        payload = {k: v for k, v in plan.items() if k not in {"id", "plan_sha256", "created_at"}}
        if plan.get("method") != METHOD or _digest(payload) != plan.get("plan_sha256"):
            raise ValueError("Saved landscape plan is unsupported or has failed its integrity check")
        return plan

    def _coverage(self, grid):
        matched = {}
        with self.store.connection() as db:
            # Stream saved regions rather than loading all observation histories
            # together. Work scales with saved evidence, never planned tiles.
            for row in db.execute("SELECT id,metadata,observations FROM regions ORDER BY id"):
                region = json.loads(row["metadata"])
                provenance = region.get("provenance", {})
                if provenance.get("synthetic") is not False or provenance.get("mode") in {"demo", "synthetic"}:
                    continue
                bbox = _rectangle_bbox(region.get("geometry"))
                index = grid.index_for(bbox) if bbox else None
                if index is None:
                    continue
                observations = json.loads(row["observations"])
                dates = {item["date"] for item in observations
                         if isinstance(item.get("date"), str) and _number(item.get("valid_fraction"))
                         and item["valid_fraction"] > 0
                         and any(_number(item.get(key)) for key in MEASUREMENTS)}
                if not dates:
                    continue
                evidence = {"region_id": row["id"], "observation_count": len(dates),
                            "latest_date": max(dates)}
                previous = matched.get(index)
                # Duplicate imported/acquired rectangles are alternatives, not
                # additional monitored area or implicitly merged time series.
                if previous is None or (evidence["observation_count"], evidence["latest_date"], evidence["region_id"]) > (
                        previous["observation_count"], previous["latest_date"], previous["region_id"]):
                    matched[index] = evidence
        return matched

    def page(self, landscape_id, offset=0, limit=50):
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("Offset must be a nonnegative integer")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_PAGE:
            raise ValueError(f"Page size must be between 1 and {MAX_PAGE}")
        plan = self.get(landscape_id)
        grid = TileGrid(plan["bbox"], plan["tile_area_km2"])
        matched = self._coverage(grid)
        tiles = []
        for index in range(offset, min(offset + limit, grid.tile_count)):
            tile = grid.tile(index)
            suffix = f" · tile {index + 1}"
            tile.update(name=plan["name"][:120 - len(suffix)] + suffix,
                        region_id=None, observation_count=0, latest_date=None)
            tile.update(matched.get(index, {}))
            tiles.append(tile)
        observed_area = math.fsum(grid.tile(index)["area_km2"] for index in matched)
        return {"landscape": plan, "tiles": tiles,
                "next_offset": offset + len(tiles) if offset + len(tiles) < grid.tile_count else None,
                "summary": {"tiles_with_observations": len(matched), "total_tiles": grid.tile_count,
                            "tiles_without_observations": grid.tile_count - len(matched),
                            "study_area_with_observations_km2": observed_area,
                            "study_area_km2": grid.area_km2, "note": NOTE}}

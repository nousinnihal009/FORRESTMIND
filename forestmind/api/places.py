"""Explicit, cached worldwide place lookup; names are not forest boundaries."""
from __future__ import annotations

import json
import math
import os
import threading
import time
from urllib.error import URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler

from fastapi import HTTPException, Query

ATTRIBUTION = "© OpenStreetMap contributors · Photon / komoot"
SOURCE_URL = "https://www.openstreetmap.org/copyright"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class PlaceSearch:
    def __init__(self, store):
        self.store = store
        self.lock = threading.Lock()
        self.last_request = 0.0
        # Operator-selected HTTPS Photon-compatible service, never browser input.
        self.endpoint = os.environ.get("FORESTMIND_GEOCODER_URL", "https://photon.komoot.io/api/")
        parsed = urlsplit(self.endpoint)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query:
            raise ValueError("FORESTMIND_GEOCODER_URL must be an HTTPS endpoint without credentials or query")
        with store.connection() as db:
            db.execute("CREATE TABLE IF NOT EXISTS place_cache (query TEXT PRIMARY KEY, fetched REAL NOT NULL, payload TEXT NOT NULL)")

    def _fetch(self, query):
        request = Request(self.endpoint + "?" + urlencode({"q": query, "limit": 8, "lang": "en"}),
                          headers={"User-Agent": "ForestMind/0.6 (local forest monitoring research)", "Accept": "application/json"})
        with build_opener(NoRedirect).open(request, timeout=15) as response:
            raw = response.read(1_000_001)
        if len(raw) > 1_000_000:
            raise ValueError("Geocoder response exceeded its size limit")
        return json.loads(raw)

    def search(self, query):
        query = " ".join(query.split())
        if len(query) < 2:
            raise HTTPException(422, "Enter at least two characters or use coordinates")
        key = self.endpoint + "|" + query.casefold()
        # One outstanding request, explicit submissions only. Persist cache across restart.
        with self.lock:
            with self.store.connection() as db:
                saved = db.execute("SELECT fetched,payload FROM place_cache WHERE query=?", (key,)).fetchone()
            if saved and time.time() - saved["fetched"] < 7 * 86400:
                return {**json.loads(saved["payload"]), "cached": True}
            if time.monotonic() - self.last_request < 1.1:
                raise HTTPException(429, "Please wait a second before another place search", headers={"Retry-After": "2"})
            self.last_request = time.monotonic()
            try:
                payload = self._fetch(query)
                features = payload.get("features")
                if not isinstance(features, list):
                    raise ValueError("Invalid geocoder response")
                results = []
                for feature in features[:8]:
                    try:
                        p = feature["properties"]
                        lon, lat = map(float, feature["geometry"]["coordinates"])
                        if not math.isfinite(lon + lat) or not (-180 <= lon <= 180 and -90 <= lat <= 90):
                            continue
                        extent = p.get("extent")  # Photon extent: west,north,east,south.
                        bbox = [float(extent[i]) for i in (0, 3, 2, 1)] if extent and len(extent) == 4 else None
                        if bbox and (not all(math.isfinite(v) for v in bbox) or not (-180 <= bbox[0] <= bbox[2] <= 180 and -90 <= bbox[1] <= bbox[3] <= 90)):
                            bbox = None
                        label = ", ".join(dict.fromkeys(str(p[k]) for k in ("name", "city", "state", "country") if p.get(k)))[:500]
                        results.append({"id": f"{p.get('osm_type','place')}-{p.get('osm_id',len(results))}",
                                        "label": label or query, "country": str(p.get("country", "Unspecified"))[:100],
                                        "center": [lon, lat], "bbox": bbox, "source": ATTRIBUTION,
                                        "source_url": SOURCE_URL, "verified_forest_boundary": False,
                                        "category": str(p.get("osm_value", "named place"))[:100]})
                    except (KeyError, TypeError, ValueError):
                        continue
            except (URLError, OSError, ValueError, TypeError):
                raise HTTPException(503, "Place search is unavailable. Retry later, or select an area using coordinates or the map.") from None
            result = {"results": results, "source": ATTRIBUTION, "source_url": SOURCE_URL,
                      "cached": False, "boundary_note": "Named places are search references, not verified forest boundaries."}
            with self.store.connection() as db:
                db.execute("INSERT OR REPLACE INTO place_cache VALUES (?,?,?)", (key, time.time(), self.store._json(result)))
                db.execute("DELETE FROM place_cache WHERE query IN (SELECT query FROM place_cache ORDER BY fetched DESC LIMIT -1 OFFSET 1000)")
            return result


def install_places(app, store):
    service = PlaceSearch(store)
    app.state.places = service

    @app.get("/api/v1/places")
    def places(q: str = Query(min_length=2, max_length=200)):
        return service.search(q)

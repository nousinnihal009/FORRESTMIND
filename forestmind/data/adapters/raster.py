"""Read every satellite band onto the same explicitly georeferenced AOI grid."""

from __future__ import annotations

import numpy as np

from forestmind.utils.geo import BoundingBox


def read_aligned_band(src, bbox: BoundingBox, patch_size: int,
                      categorical: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Reproject a raster into the exact WGS84 AOI, returning values and validity.

    WarpedVRT transforms the destination grid into the native source CRS before
    reading. Passing longitude/latitude directly into a UTM raster window would
    instead select an unrelated or empty region. A single destination affine
    also aligns differently projected scenes and differently resolved bands.

    Resolution follows AOI extent / patch_size, in degrees. It is not implicitly
    the native 10/20/30/60 metre sensor resolution. Categorical QA uses nearest
    neighbour resampling so class IDs and bit flags remain intact.
    """
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.vrt import WarpedVRT

    if src.crs is None:
        raise ValueError("Satellite raster has no CRS; geographic alignment is unavailable")
    if patch_size < 1 or bbox.west >= bbox.east or bbox.south >= bbox.north:
        raise ValueError("Target grid requires a positive patch size and ordered bbox bounds")
    transform = from_bounds(bbox.west, bbox.south, bbox.east, bbox.north,
                            patch_size, patch_size)
    with WarpedVRT(
        src, crs="EPSG:4326", transform=transform,
        width=patch_size, height=patch_size, dtype="float32",
        nodata=np.nan,
        resampling=Resampling.nearest if categorical else Resampling.bilinear,
    ) as aligned:
        values = aligned.read(1, masked=True)
    valid = ~np.ma.getmaskarray(values) & np.isfinite(values.data)
    return np.where(valid, values.data, 0).astype(np.float32), valid

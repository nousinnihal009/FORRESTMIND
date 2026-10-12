"""Numerical raster regression tests; install the real-data extra to run these."""

import numpy as np
import pytest

from forestmind.data.adapters.raster import read_aligned_band
from forestmind.utils.geo import BoundingBox


@pytest.mark.parametrize("categorical", [False, True])
def test_projected_raster_is_aligned_to_geographic_aoi(categorical):
    pytest.importorskip("rasterio", reason="Optional real-data rasterio dependency is not installed")
    from rasterio.io import MemoryFile
    from rasterio.transform import from_bounds
    from rasterio.warp import transform_bounds

    bbox = BoundingBox(west=-1, south=-1, east=1, north=1)
    native_bounds = transform_bounds("EPSG:4326", "EPSG:3857", -1, -1, 1, 1)
    source = np.full((16, 16), 4., dtype=np.float32)
    source[:, :8] = 1.
    source[:4, :4] = -9999.
    with MemoryFile() as memory:
        with memory.open(driver="GTiff", width=16, height=16, count=1,
                         crs="EPSG:3857", transform=from_bounds(*native_bounds, 16, 16),
                         dtype="float32", nodata=-9999.) as raster:
            raster.write(source, 1)
        with memory.open() as raster:
            values, valid = read_aligned_band(raster, bbox, 32, categorical=categorical)
    assert values.shape == valid.shape == (32, 32)
    assert values[16, 4] == pytest.approx(1.)
    assert values[16, 28] == pytest.approx(4.)
    assert not valid[1, 1]
    assert valid[16, 16]
    if categorical:
        assert set(np.unique(values[valid])) == {1., 4.}


def test_raster_without_crs_is_rejected():
    pytest.importorskip("rasterio", reason="Optional real-data rasterio dependency is not installed")
    from types import SimpleNamespace
    with pytest.raises(ValueError, match="no CRS"):
        read_aligned_band(SimpleNamespace(crs=None), BoundingBox(1, 2, 3, 4), 32)

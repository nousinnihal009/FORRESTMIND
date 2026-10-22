"""Measured features, sensor masks and missingness survive the neural path."""
import numpy as np
import pytest
import torch

from forestmind.data.transforms import Normalize, SpatialFlipRotate
from forestmind.models.fusion.multimodal import MultimodalFusion


@pytest.mark.parametrize("method", ["cross_attention", "gated", "concat"])
def test_missing_context_is_identity_and_masked_values_cannot_change_output(method):
    torch.manual_seed(14)
    model = MultimodalFusion(embed_dim=8, method=method, num_heads=2, dropout=0,
                             climate_dim=4).eval()
    primary = torch.randn(3, 8)
    climate = torch.randn(3, 4)
    mask = torch.tensor([[False]*4, [True, False, True, False], [True]*4])
    first = model(primary, climate_context=climate, climate_mask=mask)
    changed = climate.clone()
    changed[~mask] = float("nan")
    second = model(primary, climate_context=changed, climate_mask=mask)
    assert torch.equal(first[0], primary[0])
    assert torch.equal(first, second)
    assert not torch.equal(first[1:], primary[1:])
    with pytest.raises(ValueError, match="explicit feature mask"):
        model(primary, climate_context=climate)


def test_normalization_masks_independent_radar_pixels_and_climate_features():
    sample = {"optical": np.ones((1, 12, 2, 2), np.float32),
              "sar": np.ones((1, 2, 2, 2), np.float32),
              "sar_quality_mask": np.array([[[True, False], [False, True]]]),
              "climate_context": np.array([[20., 999., 4., 999.]], np.float32),
              "climate_mask": np.array([[True, False, True, False]])}
    normalized = Normalize(climate_mean=[10, 2, 2, 0], climate_std=[5, 1, 1, 1])(sample)
    assert normalized["climate_context"].tolist() == [[2., 0., 2., 0.]]
    assert not normalized["sar"][:, :, 0, 1].any()


def test_spatial_augmentation_preserves_radar_mask_alignment():
    mask = np.array([[[True, False], [False, False]]])
    sample = {"optical": np.zeros((1,12,2,2)), "sar": np.repeat(mask[:, None], 2, axis=1).astype(float),
              "sar_quality_mask": mask}
    transformed = SpatialFlipRotate()(sample)
    assert np.array_equal(transformed["sar"][:, 0] > 0, transformed["sar_quality_mask"])

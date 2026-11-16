"""Комбинации не используют другие строки и корректны на плоских изображениях."""
import numpy as np
import pandas as pd
import pytest

from main import prepare_feature_table
from src.combinations import add_combinations
from src.features import image_features


@pytest.fixture
def table():
    image = np.random.default_rng(7).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    return pd.DataFrame([image_features(image), image_features(np.full_like(image, 128))],
                        index=[7, 3]).assign(img_name=["a_1.png", "b_2.png"])


def test_combinations_preserve_base_and_are_finite_for_constant_image(table):
    before = table.copy(deep=True)
    combined = add_combinations(table)
    pd.testing.assert_frame_equal(table, before)
    pd.testing.assert_frame_equal(combined[table.columns], table)
    added = combined.filter(like="combo_")
    assert added.shape == (2, 12)
    assert np.isfinite(added.to_numpy()).all()
    assert combined.loc[3, "combo_laplacian_contrast_s1"] == 0
    assert combined.loc[7, "combo_hist_empty_range"] == pytest.approx(
        table.loc[7, "dist_hist_empty"] * (table.loc[7, "color_luma_p99"] - table.loc[7, "color_luma_p01"]) / 255)


def test_combinations_do_not_depend_on_other_rows_or_their_order(table):
    combined = add_combinations(table)
    pd.testing.assert_frame_equal(add_combinations(table.iloc[::-1]), combined.iloc[::-1])
    pd.testing.assert_frame_equal(add_combinations(table.iloc[:1]), combined.iloc[:1])


def test_combinations_reject_missing_or_invalid_inputs(table):
    with pytest.raises(ValueError, match="не хватает"):
        add_combinations(table.drop(columns="dist_noise_sigma"))
    table.loc[7, "dist_noise_sigma"] = -1
    with pytest.raises(ValueError, match="неотрицательные"):
        add_combinations(table)


def test_prepare_feature_table_only_adds_combinations_when_enabled(tmp_path, table):
    path = tmp_path / "features.csv"
    table.to_csv(path, index=False)
    config = {"table": path, "recompute": False}
    names = table["img_name"]
    original = prepare_feature_table(config, names, "img_name")
    combined = prepare_feature_table({**config, "combinations": True}, names, "img_name")
    assert combined.shape[1] == original.shape[1] + 12
    pd.testing.assert_frame_equal(combined[original.columns], original)
    assert pd.read_csv(path).shape == original.shape
    with pytest.raises(ValueError, match="true или false"):
        prepare_feature_table({**config, "combinations": "false"}, names, "img_name")

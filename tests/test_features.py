"""Проверки признаков на синтетических изображениях с известными искажениями."""
import cv2
import numpy as np
import pandas as pd
import pytest

from main import prepare_feature_table
from src.features import extract_features, find_images, image_features

GROUPS = ("color", "sharp", "noise", "comp", "spectrum", "nss", "texture")
N_FEATURES = 141


@pytest.fixture(scope="module")
def base_image():
    """Создаёт плавное цветное изображение с полным диапазоном яркости."""
    rng = np.random.default_rng(0)
    image = cv2.GaussianBlur(rng.integers(0, 256, (480, 640, 3), dtype=np.uint8), (0, 0), 3)
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)


@pytest.fixture(scope="module")
def clean(base_image):
    return image_features(base_image)


def test_features_depend_only_on_pixels_and_have_known_groups(clean):
    assert len(clean) == N_FEATURES
    assert {name.split("_")[0] for name in clean} == set(GROUPS)


def test_blur_lowers_sharpness_and_raises_reblur_ratio(base_image, clean):
    blurred = image_features(cv2.GaussianBlur(base_image, (0, 0), 5))
    assert blurred["sharp_s1_laplacian_var"] < clean["sharp_s1_laplacian_var"]
    assert blurred["sharp_reblur_vertical"] > clean["sharp_reblur_vertical"]


def test_noise_raises_noise_estimate(base_image, clean):
    rng = np.random.default_rng(0)
    noisy = np.clip(base_image + rng.normal(0, 25, base_image.shape), 0, 255).astype(np.uint8)
    assert image_features(noisy)["noise_immerkaer_gray"] > clean["noise_immerkaer_gray"]


def test_compression_raises_blockiness(base_image, clean):
    encoded = cv2.imencode(".jpg", base_image, [cv2.IMWRITE_JPEG_QUALITY, 10])[1]
    compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    assert image_features(compressed)["comp_block8_horizontal"] > clean["comp_block8_horizontal"]


def test_constant_image_gives_finite_features(base_image, clean):
    features = image_features(np.full_like(base_image, 128))
    assert features.keys() == clean.keys()
    assert np.isfinite(list(features.values())).all()


def test_find_images_reports_missing_file(tmp_path):
    with pytest.raises(ValueError, match="нет 1 изображений"):
        find_images(["absent_1.png"], tmp_path)


def write_images(images_dir, pictures):
    """Сохраняет изображения в PNG; cv2.imwrite в Windows не поддерживает пути с не-ASCII символами."""
    images_dir.mkdir()
    for name, picture in pictures.items():
        cv2.imencode(".png", picture)[1].tofile(images_dir / name)


def test_extract_features_keeps_name_order_and_reads_non_ascii_path(tmp_path, base_image, clean):
    images_dir = tmp_path / "изображения"
    write_images(images_dir, {"a_1.png": base_image, "b_2.png": 255 - base_image})
    names = ["b_2.png", "a_1.png"]

    table = extract_features(names, images_dir, "img_name")

    assert table["img_name"].tolist() == names
    assert table.loc[1, "color_red_mean"] == pytest.approx(clean["color_red_mean"])
    assert table.shape == (2, N_FEATURES + 1)


def test_prepare_feature_table_recomputes_and_saves_table(tmp_path, base_image, clean):
    images_dir = tmp_path / "images"
    write_images(images_dir, {"a_1.png": base_image})
    path = tmp_path / "new" / "features.csv.gz"
    config = {"table": path, "recompute": True, "images_dir": images_dir}

    prepare_feature_table(config, ["a_1.png"], "img_name")

    saved = pd.read_csv(path)
    assert saved["img_name"].tolist() == ["a_1.png"]
    assert saved.loc[0, "color_red_mean"] == pytest.approx(clean["color_red_mean"], rel=1e-6)
    assert saved.shape == (1, N_FEATURES + 1)

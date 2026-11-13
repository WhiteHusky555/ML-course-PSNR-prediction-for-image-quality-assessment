"""Проверки свёрточной нейросети на изображениях."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from main import run
from src.cnn import ImageCNNRegressor
from src.models import model_columns

SMALL = {"backbone": "small", "filters": [4, 8], "batch_size": 8}


def write_images(directory, names, brightness, size=(16, 24)):
    """Сохраняет однотонные изображения заданного размера и яркости."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in zip(names, brightness):
        Image.fromarray(np.full((*size, 3), value, dtype=np.uint8)).save(directory / name)


def test_cnn_learns_brightness(tmp_path):
    names = [f"{index}_1.png" for index in range(48)]
    brightness = np.linspace(20, 235, len(names))
    write_images(tmp_path, names, brightness.astype(np.uint8))
    target = 25 + brightness / 50
    model = ImageCNNRegressor(images_dir=tmp_path, epochs=60, learning_rate=0.01, **SMALL)
    predicted = model.fit(pd.DataFrame({"img_name": names}), target).predict(pd.DataFrame({"img_name": names}))
    assert np.abs(predicted - target).mean() < 0.5 * np.abs(target - target.mean()).mean()


# Вырезка заметно меньше изображения: обучение на фрагментах, прогноз по целому кадру.
@pytest.mark.parametrize("size, options", [((32, 48), {"crop": [16, 16]}), ((16, 24), {"mixed_precision": True})])
def test_cnn_learns_brightness_with_crops_or_mixed_precision(tmp_path, size, options):
    names = [f"{index}_1.png" for index in range(48)]
    brightness = np.linspace(20, 235, len(names))
    write_images(tmp_path, names, brightness.astype(np.uint8), size)
    target = 25 + brightness / 50
    model = ImageCNNRegressor(images_dir=tmp_path, epochs=60, learning_rate=0.01, **SMALL, **options)
    predicted = model.fit(pd.DataFrame({"img_name": names}), target).predict(pd.DataFrame({"img_name": names}))
    assert predicted.dtype == np.float32
    assert np.abs(predicted - target).mean() < 0.5 * np.abs(target - target.mean()).mean()


def test_mixed_precision_does_not_change_global_policy(tmp_path):
    import keras

    write_images(tmp_path, ["a_1.png", "b_1.png"], [100, 200])
    ImageCNNRegressor(images_dir=tmp_path, epochs=1, mixed_precision=True, **SMALL).fit(
        pd.DataFrame({"img_name": ["a_1.png", "b_1.png"]}), [28.0, 30.0])
    assert keras.config.dtype_policy().name == "float32"


def test_cnn_rejects_unknown_backbone(tmp_path):
    write_images(tmp_path, ["a_1.png"], [100])
    with pytest.raises(ValueError):
        ImageCNNRegressor(images_dir=tmp_path, backbone="vit", epochs=1).fit(pd.DataFrame({"img_name": ["a_1.png"]}), [28.0])


def test_image_model_receives_only_file_names():
    assert model_columns({"type": "cnn"}, ["blur", "noise"]) == ["img_name"]


def test_run_trains_cnn_on_images(config):
    names = pd.concat([pd.read_csv(config["data"]["dir"] / file)["img_name"]
                       for file in ("train.csv", "sample_submission.csv")])
    write_images(config["features"]["images_dir"] / "images", names, np.full(len(names), 128))
    config["models"] = {"cnn": {"type": "cnn", "params": {**SMALL, "epochs": 1}}}

    results = run(config)

    assert results["model"].tolist() == ["cnn"]
    assert len(pd.read_csv(config["output"]["dir"] / "submission.csv")) == 20

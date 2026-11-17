"""Сравнение стекеров на сохранённых прогнозах для 155 и 167 признаков, без обучения базовых моделей."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from main import attach_predictions, config_folds, run
from src.config import load_config
from src.data import check_columns, check_target, load_data
from src.models import build_model, expand_grid, model_columns


def evaluate_arm(arm, output):
    """Проверяет сохранённые прогнозы, выполняет общий пайплайн и сохраняет коэффициенты фолдов."""
    config = load_config(ROOT / "experiments/ensemble_linear.yaml")
    for pair in config["predictions"]:
        for key, value in pair.items():
            pair[key] = str(ROOT / value.replace("outputs/baseline/", f"outputs/{arm}/"))
    hashes = {}
    reference_train, reference_test = None, None
    for pair in config["predictions"]:
        train = pd.read_csv(pair["oof"])
        test = pd.read_csv(pair["test"])
        check_columns(train, ["img_name", "PSNR"], pair["oof"])
        check_columns(test, ["img_name"], pair["test"])
        check_target(train, "PSNR")
        models = [c for c in test if c != "img_name"]
        if set(train.columns) != {"img_name", "PSNR", *models}:
            raise ValueError("Состав моделей OOF и test не совпадает")
        for frame in (train, test):
            if not np.isfinite(frame[models].to_numpy(dtype=float)).all():
                raise ValueError("Неконечные прогнозы базовых моделей")
        if reference_train is None:
            reference_train, reference_test = train, test
        else:
            if set(train.img_name) != set(reference_train.img_name) or set(test.img_name) != set(reference_test.img_name):
                raise ValueError("Источники прогнозов покрывают разные изображения")
            aligned = train.set_index("img_name").loc[reference_train.img_name, "PSNR"].to_numpy()
            np.testing.assert_allclose(aligned, reference_train.PSNR, rtol=0, atol=1e-10)
        for path in pair.values():
            hashes[str(Path(path).relative_to(ROOT))] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if len(reference_train) != 13210 or len(reference_test) != 3310:
        raise ValueError("Ожидались 13210 строк train и 3310 test")
    data = output / "data"
    data.mkdir(parents=True, exist_ok=True)
    reference_train[["img_name", "PSNR"]].to_csv(data / "train.csv", index=False)
    reference_test[["img_name"]].to_csv(data / "sample_submission.csv", index=False)
    config["data"]["dir"] = str(data)
    config["output"]["dir"] = str(output)
    config["features"]["table"] = str(ROOT / config["features"]["table"])
    config["features"]["combinations"] = arm == "combinations"
    # Четыре потока соответствуют числу CPU прежних прогонов и не перегружают локальную машину.
    for spec in config["models"].values():
        if spec["type"] == "lightgbm":
            spec["params"]["n_jobs"] = 4
    results = run(config)
    train, test = load_data(config["data"])
    train, test = attach_predictions(train, test, config["predictions"], "img_name")
    folds = config_folds(config, train, "PSNR", config["validation"]["seed"])
    oof = pd.read_csv(output / "oof_predictions.csv")
    test_predictions = pd.read_csv(output / "test_predictions.csv")
    coefficients, fold_rows = [], []
    for name, spec in expand_grid(config["models"]).items():
        errors = (oof.PSNR - oof[name]).abs()
        score = results.set_index("model").loc[name]
        np.testing.assert_allclose(errors.mean(), score.mae, rtol=0, atol=1e-10)
        for number, (_, valid) in enumerate(folds, 1):
            fold_rows.append({"model": name, "fold": number, "mae": errors.iloc[valid].mean()})
        if spec["type"] not in ("linear", "ridge"):
            continue
        columns = model_columns(spec, train.columns)
        fold_test = []
        for number, (fit, valid) in enumerate(folds, 1):
            model = build_model(spec).fit(train.iloc[fit][columns], train.PSNR.iloc[fit])
            # Проверка: сохранённые коэффициенты воспроизводят OOF и средний прогноз на test.
            predicted = train.iloc[valid][columns].to_numpy() @ model.coef_ + model.intercept_
            np.testing.assert_allclose(predicted, oof[name].iloc[valid], rtol=0, atol=1e-9)
            fold_test.append(test[columns].to_numpy() @ model.coef_ + model.intercept_)
            coefficients.append({"model": name, "fold": number, "intercept": model.intercept_,
                                 **dict(zip(columns, model.coef_))})
        np.testing.assert_allclose(np.mean(fold_test, axis=0), test_predictions[name], rtol=0, atol=1e-9)
    pd.DataFrame(coefficients).to_csv(output / "coefficients.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_scores.csv", index=False)
    (output / "input_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    return results.assign(features=167 if arm == "combinations" else 155)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/ensemble_linear")
    args = parser.parse_args()
    started = time.perf_counter()
    results = []
    for arm in ("baseline", "combinations"):
        print(f"== {arm}", flush=True)
        results.append(evaluate_arm(arm, args.output / arm))
    combined = pd.concat(results, ignore_index=True)
    combined.to_csv(args.output / "comparison.csv", index=False)
    versions = {"python": platform.python_version(), "platform": platform.platform(),
                "seconds": time.perf_counter() - started}
    versions.update({p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy", "lightgbm")})
    (args.output / "versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    print(combined[["features", "model", "mae", "fold_std"]].to_string(index=False), flush=True)

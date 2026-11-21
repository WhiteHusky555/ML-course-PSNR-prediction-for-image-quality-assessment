"""Проверка VotingRegressor на 155 признаках и пяти групповых фолдах."""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.preprocessing import QuantileClipper
from src.validation import make_folds


def run(output):
    """Обучает участников внутри фолдов и проверяет равенство прогнозов."""
    config_path = Path(__file__).with_name("config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paths = [ROOT / config[key] for key in ("features", "train", "test")]
    features = pd.read_csv(paths[0])
    train = pd.read_csv(paths[1])[["img_name", "PSNR"]]
    test = pd.read_csv(paths[2])[["img_name"]]
    for frame in (features, train, test):
        if frame.img_name.isna().any() or frame.img_name.duplicated().any():
            raise ValueError("Пустые или повторяющиеся имена")
    if len(train) != 13210 or len(test) != 3310:
        raise ValueError("Неожиданный размер данных")
    columns = [c for c in features if c != "img_name"]
    if len(columns) != 155:
        raise ValueError("Ожидались 155 признаков")
    train = train.merge(features, on="img_name", how="left", validate="one_to_one")
    test = test.merge(features, on="img_name", how="left", validate="one_to_one")
    if not np.isfinite(train[columns + ["PSNR"]].to_numpy()).all() or not np.isfinite(test[columns].to_numpy()).all():
        raise ValueError("Пропущенные или неконечные значения")
    train["source"] = train.img_name.str.rsplit("_", n=1).str[0]
    test_sources = test.img_name.str.rsplit("_", n=1).str[0]
    if set(train.source) & set(test_sources):
        raise ValueError("Пересечение исходников train/test")
    folds = make_folds(train, "PSNR", **config["validation"])
    names = ["ridge", "tree", "histgb", "voting", "manual_mean"]
    oof = np.zeros((len(train), len(names)))
    predictions = np.zeros((len(test), len(names)))
    fold_ids = np.full(len(train), -1)
    scores = []
    max_difference = 0.0
    for fold, (fit_idx, valid_idx) in enumerate(folds):
        if set(train.source.iloc[fit_idx]) & set(train.source.iloc[valid_idx]):
            raise ValueError("Пересечение групп фолда")
        model = VotingRegressor([
            ("ridge", make_pipeline(QuantileClipper(0.001, 0.999), StandardScaler(), Ridge(**config["ridge"]))),
            ("tree", DecisionTreeRegressor(**config["tree"])),
            ("histgb", HistGradientBoostingRegressor(**config["histgb"])),
        ], weights=None, n_jobs=1)
        model.fit(train.iloc[fit_idx][columns], train.PSNR.iloc[fit_idx])
        arrays = []
        for frame in (train.iloc[valid_idx], test):
            members = np.column_stack([est.predict(frame[columns]) for est in model.estimators_])
            voting = model.predict(frame[columns])
            manual = members.mean(axis=1)
            np.testing.assert_allclose(voting, manual, rtol=0, atol=1e-12)
            max_difference = max(max_difference, float(np.max(np.abs(voting - manual))))
            arrays.append(np.column_stack([members, voting, manual]))
        oof[valid_idx] = arrays[0]
        predictions += arrays[1] / len(folds)
        fold_ids[valid_idx] = fold
        for i, name in enumerate(names):
            scores.append({"fold": fold, "model": name, "mae": float(np.mean(np.abs(train.PSNR.iloc[valid_idx].to_numpy() - arrays[0][:, i])))})
        print(f"Fold {fold + 1}: Voting MAE {scores[-2]['mae']:.6f}", flush=True)
    if (fold_ids < 0).any():
        raise ValueError("Неполное покрытие OOF")
    output.mkdir(parents=True, exist_ok=True)
    fold_table = pd.DataFrame(scores)
    results = pd.DataFrame([
        {"model": name, "mae": float(np.mean(np.abs(train.PSNR.to_numpy() - oof[:, i]))),
         "fold_std": float(fold_table.loc[fold_table.model == name, "mae"].std(ddof=0))}
        for i, name in enumerate(names)
    ]).sort_values("mae")
    results.to_csv(output / "results.csv", index=False)
    fold_table.to_csv(output / "fold_scores.csv", index=False)
    train[["img_name", "PSNR"]].assign(fold=fold_ids, **dict(zip(names, oof.T))).to_csv(output / "oof_predictions.csv", index=False)
    test[["img_name"]].assign(**dict(zip(names, predictions.T))).to_csv(output / "test_predictions.csv", index=False)
    test[["img_name"]].assign(PSNR=predictions[:, 3]).to_csv(output / "submission_voting.csv", index=False)
    metadata = {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__,
                "pandas": pd.__version__, "max_voting_mean_difference": max_difference,
                "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()}
    (output / "verification.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(results.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/voting")
    run(parser.parse_args().output)

"""Сравнивает 155 признаков и 155 + 12 комбинаций на одном коммите и тех же фолдах Kaggle."""
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

REPOSITORY = "https://github.com/WhiteHusky555/ML-course-PSNR-prediction-for-image-quality-assessment.git"
COMMIT = "027872888f07a2c84f1fc2f26f6dacea5949d773"
WORKING = Path("/kaggle/working")
REPO = WORKING / "repo"
PACKAGES = ["tensorflow", "keras", "numpy", "pandas", "scikit-learn", "scipy", "lightgbm", "xgboost", "catboost"]


def find_data_dir(input_dir):
    """Проверяет подключение соревнования до установки зависимостей и обучения."""
    for train in input_dir.rglob("train.csv"):
        if (train.parent / "sample_submission.csv").is_file():
            return train.parent
    raise ValueError(
        f"В {input_dir} нет train.csv и sample_submission.csv в одной папке. "
        "Подключите источник соревнования psnr-iqa к ядру Kaggle."
    )


def run_comparison():
    """Повторяет контроль и эксперимент; исходная таблица признаков не перезаписывается."""
    if not re.fullmatch(r"[0-9a-f]{40}", COMMIT):
        raise ValueError("Сначала укажите полный хеш опубликованного коммита в COMMIT")
    data_dir = find_data_dir(Path("/kaggle/input"))
    print(f"Competition input verified: {data_dir}", flush=True)
    subprocess.run(["git", "clone", "--quiet", REPOSITORY, str(REPO)], check=True)
    subprocess.run(["git", "checkout", "--quiet", COMMIT], cwd=REPO, check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-r", "requirements.txt"],
                   cwd=REPO, check=True)
    versions = {"python": platform.python_version(), "commit": COMMIT}
    versions.update({name: importlib.metadata.version(name) for name in PACKAGES})
    (WORKING / "versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    print(json.dumps(versions), flush=True)
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=REPO, check=True)
    for label, enabled in (("baseline", False), ("combinations", True)):
        for stage in ("ensemble_base", "ensemble"):
            name = "combinations" if enabled and stage == "ensemble_base" else stage
            config = yaml.safe_load((REPO / f"experiments/{name}.yaml").read_text(encoding="utf-8"))
            config["data"]["dir"] = str(data_dir)
            config["features"].update(table=str(REPO / "features/features.csv.gz"), recompute=False,
                                      combinations=enabled)
            config["output"]["dir"] = str(WORKING / "outputs" / label / stage)
            for pair in config.get("predictions", []):
                for key, value in pair.items():
                    pair[key] = str(WORKING / "outputs" / label / "ensemble_base" / Path(value).name
                                    if value.startswith("outputs/ensemble_base/") else REPO / value)
            path = WORKING / f"run_{label}_{stage}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            print(f"== {label}: {stage}", flush=True)
            subprocess.run([sys.executable, "main.py", "--config", str(path)], cwd=REPO, check=True)
    comparisons = []
    for stage in ("ensemble_base", "ensemble"):
        before = pd.read_csv(WORKING / "outputs/baseline" / stage / "results.csv")
        after = pd.read_csv(WORKING / "outputs/combinations" / stage / "results.csv")
        paired = before.merge(after, on="model", suffixes=("_before", "_after"), validate="one_to_one")
        if len(paired) != len(before) or len(paired) != len(after):
            raise ValueError("Состав моделей контроля и эксперимента различается")
        paired["delta_mae"] = paired["mae_after"] - paired["mae_before"]
        paired.insert(0, "stage", stage)
        comparisons.append(paired)
    summary = pd.concat(comparisons, ignore_index=True)
    summary.to_csv(WORKING / "comparison.csv", index=False)
    print(summary[["stage", "model", "mae_before", "mae_after", "delta_mae"]].to_string(index=False), flush=True)
    shutil.rmtree(REPO)


if __name__ == "__main__":
    run_comparison()

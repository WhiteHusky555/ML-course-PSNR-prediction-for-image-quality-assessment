"""Проверка входных данных до установки библиотек и обучения."""
import pytest

from run_combinations import find_data_dir


def test_missing_competition_input_has_actionable_error(tmp_path):
    with pytest.raises(ValueError, match="psnr-iqa"):
        find_data_dir(tmp_path)


def test_train_without_submission_template_is_not_ready(tmp_path):
    (tmp_path / "train.csv").write_text("img_name,PSNR\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sample_submission.csv"):
        find_data_dir(tmp_path)


def test_finds_nested_competition_input(tmp_path):
    data = tmp_path / "competitions" / "psnr-iqa"
    data.mkdir(parents=True)
    (data / "train.csv").write_text("img_name,PSNR\n", encoding="utf-8")
    (data / "sample_submission.csv").write_text("img_name,PSNR\n", encoding="utf-8")
    assert find_data_dir(tmp_path) == data

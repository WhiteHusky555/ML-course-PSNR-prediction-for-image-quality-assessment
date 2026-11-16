"""Фиксированные комбинации пиксельных признаков без обучения и использования меток."""
import numpy as np


def add_combinations(table):
    """Добавляет 12 отношений и взаимодействий; исходные столбцы и порядок строк сохраняются.

    Единица в знаменателях задаёт нижний масштаб контраста/энергии, 0,1 — шума.
    Это ограничивает отношения на плоских и почти постоянных изображениях.
    """
    required = [
        "color_luma_std", "color_luma_p01", "color_luma_p99",
        "sharp_s1_laplacian_var", "sharp_s2_laplacian_var", "sharp_s4_laplacian_var",
        "sharp_s1_gradient_p50", "sharp_s1_gradient_p99", "dist_noise_sigma",
        "noise_immerkaer_red", "noise_immerkaer_green", "noise_immerkaer_blue",
        "comp_dct_high_mean", "comp_dct_low_mean", "dist_hist_empty", "dist_hist_roughness",
        *(f"spectrum_ring{i}_share" for i in range(8)),
    ]
    missing = [name for name in required if name not in table.columns]
    if missing:
        raise ValueError(f"Для комбинаций не хватает признаков: {missing}")
    values = table[required].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Комбинациям нужны конечные неотрицательные исходные признаки")
    if (table["color_luma_p99"] < table["color_luma_p01"]).any():
        raise ValueError("Квантиль p99 не может быть меньше p01")

    contrast = table["color_luma_std"]
    dynamic_range = (table["color_luma_p99"] - table["color_luma_p01"]) / 255
    laplacians = [table[f"sharp_s{s}_laplacian_var"] for s in (1, 2, 4)]
    noise_rgb = table[[f"noise_immerkaer_{c}" for c in ("red", "green", "blue")]]
    noise_rms = np.sqrt((noise_rgb**2).mean(axis=1))
    features = {
        f"combo_laplacian_contrast_s{s}": laplacian / (contrast**2 + 1)
        for s, laplacian in zip((1, 2, 4), laplacians)
    }
    features.update({
        "combo_laplacian_s1_s2": laplacians[0] / (laplacians[1] + 1),
        "combo_laplacian_s2_s4": laplacians[1] / (laplacians[2] + 1),
        "combo_noise_contrast": table["dist_noise_sigma"] / (contrast + 1),
        "combo_noise_haar_immerkaer": table["dist_noise_sigma"] / (noise_rms + 0.1),
        "combo_gradient_tail": table["sharp_s1_gradient_p99"] / (table["sharp_s1_gradient_p50"] + 1),
        "combo_spectrum_high_low": table[[f"spectrum_ring{i}_share" for i in (5, 6, 7)]].sum(axis=1)
                                   / (table[[f"spectrum_ring{i}_share" for i in (0, 1, 2)]].sum(axis=1) + 1e-8),
        "combo_dct_high_low": table["comp_dct_high_mean"] / (table["comp_dct_low_mean"] + 1),
        "combo_hist_empty_range": table["dist_hist_empty"] * dynamic_range,
        "combo_hist_roughness_range": table["dist_hist_roughness"] * dynamic_range,
    })
    return table.assign(**features)

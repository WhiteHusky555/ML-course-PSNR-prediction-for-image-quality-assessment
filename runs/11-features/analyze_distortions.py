"""Как получены варианты искажения: проверка гипотез JPEG, изменения размера и линейного преобразования яркости.

Для выборки исходников train загружаются все пять вариантов. По каждому изображению считаются:
* PSNR между изображением и его повторным JPEG-сжатием с качеством q: повторное сжатие с тем же
  качеством почти не меняет изображение, поэтому пик по q указывает на исходное качество;
* то же для уменьшения в s раз и обратного увеличения разными интерполяциями;
* периодичность второй производной вдоль строк и столбцов: после увеличения размера в s раз
  в ней появляется пик на частоте 1/s;
* доля пустых бинов и пиков гистограммы каналов внутри диапазона значений;
* блочность на периоде 8.
Связь вариантов одного исходника: среднее `_2` и `_3` служит оценкой оригинала (шумы независимы,
дисперсия шума в среднем вдвое меньше). Для каждого варианта считаются линейная модель
`вариант ≈ a·оценка + b` по каналам и PSNR относительно оценки с поправкой на её шум.
"""
import io
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

INPUT_DIR = Path("/kaggle/input")
OUTPUT_DIR = Path("/kaggle/working")
N_SOURCES = 200
SEED = 0
VARIANTS = [1, 2, 3, 4, 5]
JPEG_QUALITIES = list(range(10, 100, 5))
SCALES = [1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
PIL_KERNELS = {"pil_bilinear": Image.BILINEAR, "pil_bicubic": Image.BICUBIC, "pil_lanczos": Image.LANCZOS}
CV2_KERNELS = {} if cv2 is None else {"cv2_linear": cv2.INTER_LINEAR, "cv2_cubic": cv2.INTER_CUBIC,
                                      "cv2_area_linear": (cv2.INTER_AREA, cv2.INTER_LINEAR)}
PSNR_CAP = 100.0
PEAK_RATIO = 1.5


def psnr(first, second):
    mse = np.mean((first.astype(np.float64) - second.astype(np.float64)) ** 2)
    return PSNR_CAP if mse == 0 else 10 * np.log10(255.0**2 / mse)


def jpeg(rgb, quality):
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=quality)
    return np.asarray(Image.open(buffer).convert("RGB"))


def resize_roundtrip(rgb, scale, kernel):
    height, width = rgb.shape[:2]
    small = (max(1, round(width / scale)), max(1, round(height / scale)))
    if kernel in PIL_KERNELS:
        method = PIL_KERNELS[kernel]
        image = Image.fromarray(rgb)
        return np.asarray(image.resize(small, method).resize((width, height), method))
    down, up = CV2_KERNELS[kernel] if isinstance(CV2_KERNELS[kernel], tuple) else (CV2_KERNELS[kernel],) * 2
    return cv2.resize(cv2.resize(rgb, small, interpolation=down), (width, height), interpolation=up)


def gray(rgb):
    return rgb.astype(np.float64) @ np.array([0.299, 0.587, 0.114])


def resampling_peak(image, axis):
    """Частота (циклов на пиксель) и выраженность главного пика спектра профиля второй производной."""
    second = np.abs(np.diff(image, n=2, axis=axis))
    profile = second.mean(axis=1 - axis)
    spectrum = np.abs(np.fft.rfft(profile - profile.mean()))[1:]
    frequencies = np.fft.rfftfreq(profile.size)[1:]
    top = int(np.argmax(spectrum))
    return frequencies[top], spectrum[top] / (np.median(spectrum) + 1e-12)


def histogram_stats(rgb):
    empty, peaks = [], []
    for channel in range(3):
        counts = np.bincount(rgb[..., channel].ravel(), minlength=256)
        low, high = np.flatnonzero(counts)[[0, -1]]
        inner = counts[low:high + 1].astype(np.float64)
        if inner.size < 3:
            continue
        empty.append(np.mean(inner == 0))
        neighbours = (inner[:-2] + inner[2:]) / 2
        peaks.append(np.mean(inner[1:-1] > PEAK_RATIO * neighbours + 1))
    return float(np.mean(empty)) if empty else 0.0, float(np.mean(peaks)) if peaks else 0.0


def blockiness8(image):
    columns = np.abs(np.diff(image, axis=1)).mean(axis=0)
    boundary = np.arange(columns.size) % 8 == 7
    return columns[boundary].mean() / columns[~boundary].mean()


def image_stats(rgb):
    image = gray(rgb)
    row_frequency, row_peak = resampling_peak(image, 1)
    column_frequency, column_peak = resampling_peak(image, 0)
    empty, peaks = histogram_stats(rgb)
    stats = {"row_frequency": row_frequency, "row_peak": row_peak, "column_frequency": column_frequency,
             "column_peak": column_peak, "hist_empty": empty, "hist_peaks": peaks, "block8": blockiness8(image)}
    curve = [psnr(rgb, jpeg(rgb, quality)) for quality in JPEG_QUALITIES]
    stats.update({f"jpeg_q{quality}": value for quality, value in zip(JPEG_QUALITIES, curve)})
    # PSNR растёт с качеством, поэтому признак исходного качества — выброс кривой над соседними точками.
    excess = [curve[i] - (curve[i - 1] + curve[i + 1]) / 2 for i in range(1, len(curve) - 1)]
    stats.update({"jpeg_best_quality": JPEG_QUALITIES[1 + int(np.argmax(excess))], "jpeg_excess": max(excess)})
    stats.update({f"resize_{kernel}_s{scale}": psnr(rgb, resize_roundtrip(rgb, scale, kernel))
                  for kernel in [*PIL_KERNELS, *CV2_KERNELS] for scale in SCALES})
    return stats


def relation_to_reference(rgb, reference, reference_noise):
    """Линейная модель варианта по оценке оригинала и PSNR относительно неё с поправкой на шум оценки."""
    target = rgb.astype(np.float64)
    slopes, intercepts, residuals = [], [], []
    for channel in range(3):
        x, y = reference[..., channel].ravel(), target[..., channel].ravel()
        # Шум оценки оригинала занижает наклон обычной регрессии, поэтому его дисперсия вычитается из дисперсии x.
        slope = np.cov(x, y)[0, 1] / (x.var() - reference_noise)
        intercept = y.mean() - slope * x.mean()
        slopes.append(slope)
        intercepts.append(intercept)
        residuals.append(np.std(y - (slope * x + intercept)))
    mse = np.mean((target - reference) ** 2) - reference_noise
    slope, intercept = np.mean(slopes), np.mean(intercepts)
    linear_mse = np.mean(((slope - 1) * reference + intercept) ** 2) - reference_noise * (slope - 1) ** 2
    return {"slope": slope, "intercept": intercept, "residual_std": float(np.mean(residuals)),
            "psnr_reference": 10 * np.log10(255.0**2 / max(mse, 1e-6)),
            "psnr_linear": 10 * np.log10(255.0**2 / max(linear_mse, 1e-6))}


def analyse_source(paths):
    images = {variant: np.asarray(Image.open(path).convert("RGB")) for variant, path in paths.items()}
    reference = (images[2].astype(np.float64) + images[3].astype(np.float64)) / 2
    # Дисперсия шума среднего двух вариантов: Var(n2 - n3) / 4.
    reference_noise = np.mean((images[2].astype(np.float64) - images[3].astype(np.float64)) ** 2) / 4
    rows = []
    for variant, rgb in images.items():
        row = {"img_name": paths[variant].name, "variant": variant, "reference_noise": reference_noise}
        row.update(image_stats(rgb))
        row.update(relation_to_reference(rgb, reference, reference_noise))
        rows.append(row)
    return rows


def summarise(table):
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
    by_variant = table.groupby("variant")
    base = ["PSNR", "hist_empty", "hist_peaks", "block8", "row_peak", "column_peak", "slope", "intercept",
            "residual_std", "psnr_reference", "psnr_linear"]
    print("\nМедианы по вариантам\n", by_variant[base].median().T.round(3))
    jpeg_columns = [f"jpeg_q{quality}" for quality in JPEG_QUALITIES]
    print("\nPSNR повторного JPEG-сжатия, медиана по вариантам\n", by_variant[jpeg_columns].median().T.round(2))
    print("\nКачество с наибольшим выбросом PSNR повторного сжатия\n",
          pd.crosstab(table["jpeg_best_quality"], table["variant"]))
    print("\nВеличина выброса, дБ\n", by_variant["jpeg_excess"].describe().round(2))
    resize_columns = [column for column in table if column.startswith("resize_")]
    print("\nPSNR уменьшения и обратного увеличения, медиана по вариантам\n",
          by_variant[resize_columns].median().T.round(2))
    for axis in ["row", "column"]:
        strong = table[f"{axis}_peak"] > 20
        print(f"\nЧастота сильного пика второй производной ({axis}), округлено\n",
              pd.crosstab(table.loc[strong, f"{axis}_frequency"].round(3), table.loc[strong, "variant"]))
    correlations = by_variant.apply(lambda group: pd.Series({
        "psnr_reference": group["PSNR"].corr(group["psnr_reference"], method="spearman"),
        "psnr_linear": group["PSNR"].corr(group["psnr_linear"], method="spearman"),
        "mae_reference": (group["PSNR"] - group["psnr_reference"]).abs().median(),
        "mae_linear": (group["PSNR"] - group["psnr_linear"]).abs().median(),
        "hist_empty": group["PSNR"].corr(group["hist_empty"], method="spearman"),
        "hist_peaks": group["PSNR"].corr(group["hist_peaks"], method="spearman"),
        "slope": group["PSNR"].corr(group["slope"], method="spearman"),
        "abs_intercept": group["PSNR"].corr(group["intercept"].abs(), method="spearman")}))
    print("\nСвязь с меткой PSNR внутри варианта (Спирмен; mae — медиана |метка − оценка|)\n",
          correlations.T.round(3))


def main(n_sources=N_SOURCES):
    train = pd.read_csv(next(INPUT_DIR.rglob("train.csv")))
    images = {path.name: path for path in INPUT_DIR.rglob("*.png")}
    sources = train["img_name"].str.rsplit("_", n=1).str[0].drop_duplicates()
    chosen = sources.sample(n_sources, random_state=SEED)
    jobs = [{variant: images[f"{source}_{variant}.png"] for variant in VARIANTS} for source in chosen]
    with Pool() as pool:
        rows = [row for source_rows in pool.imap(analyse_source, jobs, chunksize=4) for row in source_rows]
    table = pd.DataFrame(rows).merge(train, on="img_name", how="left", validate="one_to_one")
    table.to_csv(OUTPUT_DIR / "distortions.csv", index=False)
    summarise(table)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_SOURCES)

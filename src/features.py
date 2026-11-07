"""Извлечение признаков качества из изображений.

Каждому изображению соответствует одна строка таблицы. Признаки вычисляются только из пикселей
и описывают статистики цвета, резкость, шум, блочность, спектр, естественную статистику сцены
и текстуру.
"""
import multiprocessing
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.fft import dctn
from scipy.special import gamma
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

PROGRESS_STEP = 1000
IMAGE_TIMEOUT_SECONDS = 300
THREAD_LIMITS = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}

EPS = 1e-8
SHARPNESS_SCALES = (1, 2, 4)
FLAT_GRADIENT = 2.0
REBLUR_SIZE = 9
IMPULSE_THRESHOLD = 40
BLOCK_PERIODS = (4, 8, 16, 32)
DCT_SIZE = 8
DCT_ZERO_THRESHOLD = 1.0
RADIAL_BINS = 8
NSS_SCALES = (1, 2)
GLCM_LEVELS = 64
GLCM_DISTANCES = (1, 2, 4)
GLCM_ANGLES = (0.0, np.pi / 2)
GLCM_PROPERTIES = ("contrast", "homogeneity", "energy", "correlation")
LBP_POINTS = 8
LBP_RADIUS = 1
GABOR_WAVELENGTHS = (4, 8, 16)
GABOR_ORIENTATIONS = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
GABOR_SIZE = 31

IMMERKAER_MASK = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
GGD_SHAPES = np.arange(0.2, 10.0, 0.01)
GGD_RATIOS = gamma(1 / GGD_SHAPES) * gamma(3 / GGD_SHAPES) / gamma(2 / GGD_SHAPES) ** 2


def build_gabor_bank():
    """Создаёт банк фильтров Габора с нулевой постоянной составляющей."""
    bank = {}
    for wavelength in GABOR_WAVELENGTHS:
        for index, theta in enumerate(GABOR_ORIENTATIONS):
            kernel = cv2.getGaborKernel(
                (GABOR_SIZE, GABOR_SIZE), 0.56 * wavelength, theta, wavelength, 0.5, 0, ktype=cv2.CV_32F
            )
            bank[(wavelength, index)] = kernel - kernel.mean()
    return bank


# Параллелизм задаётся пулом процессов, внутренние потоки OpenCV ему только мешали бы.
cv2.setNumThreads(1)
GABOR_BANK = build_gabor_bank()


def distribution_stats(prefix, values):
    """Возвращает основные статистики распределения значений."""
    flat = values.ravel()
    low, median, high = np.percentile(flat, [1, 50, 99])
    is_constant = flat.std() < EPS
    return {
        f"{prefix}_mean": float(flat.mean()),
        f"{prefix}_std": float(flat.std()),
        f"{prefix}_skew": 0.0 if is_constant else float(skew(flat)),
        f"{prefix}_kurt": 0.0 if is_constant else float(kurtosis(flat)),
        f"{prefix}_p01": float(low),
        f"{prefix}_p50": float(median),
        f"{prefix}_p99": float(high),
    }


def safe_correlation(first, second):
    """Считает корреляцию Пирсона; для постоянных каналов возвращает 1."""
    if first.std() < EPS or second.std() < EPS:
        return 1.0
    return float(np.corrcoef(first.ravel(), second.ravel())[0, 1])


def color_features(bgr):
    """Статистики каналов, связь между каналами и доля насыщенных пикселей."""
    blue, green, red = (bgr[:, :, index].astype(np.float32) for index in range(3))
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    features = {}
    for name, channel in (("red", red), ("green", green), ("blue", blue)):
        features.update(distribution_stats(f"color_{name}", channel))
    features.update(distribution_stats("color_luma", ycrcb[:, :, 0]))
    features.update(
        {
            "color_cr_std": float(ycrcb[:, :, 1].std()),
            "color_cb_std": float(ycrcb[:, :, 2].std()),
            "color_corr_rg": safe_correlation(red, green),
            "color_corr_gb": safe_correlation(green, blue),
            "color_corr_rb": safe_correlation(red, blue),
            "color_share_zero": float((bgr == 0).mean()),
            "color_share_full": float((bgr == 255).mean()),
        }
    )
    return features


def sharpness_at_scale(gray, scale):
    """Меры резкости изображения, уменьшенного в заданное число раз."""
    size = (gray.shape[1] // scale, gray.shape[0] // scale)
    image = gray if scale == 1 else cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    magnitude = np.hypot(grad_x, grad_y)
    p50, p90, p99 = np.percentile(magnitude, [50, 90, 99])
    prefix = f"sharp_s{scale}"
    return {
        f"{prefix}_laplacian_var": float(cv2.Laplacian(image, cv2.CV_32F).var()),
        f"{prefix}_tenengrad": float((grad_x**2 + grad_y**2).mean()),
        f"{prefix}_brenner": float(((image[:, 2:] - image[:, :-2]) ** 2).mean()),
        f"{prefix}_gradient_p50": float(p50),
        f"{prefix}_gradient_p90": float(p90),
        f"{prefix}_gradient_p99": float(p99),
        f"{prefix}_flat_share": float((magnitude < FLAT_GRADIENT).mean()),
    }


def reblur_ratio(gray, axis):
    """Мера размытия Crete: доля перепадов, которая сохраняется после повторного размытия."""
    kernel_size = (1, REBLUR_SIZE) if axis == 0 else (REBLUR_SIZE, 1)
    original = np.abs(np.diff(gray, axis=axis))
    blurred = np.abs(np.diff(cv2.blur(gray, kernel_size), axis=axis))
    lost = np.maximum(original - blurred, 0)
    return float((original.sum() - lost.sum()) / (original.sum() + EPS))


def sharpness_features(gray):
    """Резкость на нескольких масштабах и устойчивость к повторному размытию."""
    features = {}
    for scale in SHARPNESS_SCALES:
        features.update(sharpness_at_scale(gray, scale))
    features["sharp_reblur_vertical"] = reblur_ratio(gray, 0)
    features["sharp_reblur_horizontal"] = reblur_ratio(gray, 1)
    return features


def immerkaer_sigma(channel):
    """Оценка стандартного отклонения шума методом Immerkaer."""
    height, width = channel.shape
    response = np.abs(cv2.filter2D(channel, cv2.CV_32F, IMMERKAER_MASK))[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * response.sum() / (6 * (height - 2) * (width - 2)))


def split_blocks(image, size):
    """Разбивает изображение на неперекрывающиеся квадратные блоки."""
    rows = image.shape[0] // size
    columns = image.shape[1] // size
    cropped = image[: rows * size, : columns * size]
    return cropped.reshape(rows, size, columns, size).transpose(0, 2, 1, 3).reshape(-1, size, size)


def noise_features(bgr, gray):
    """Оценки шума по всему изображению, по каналам и в однородных областях."""
    median = cv2.medianBlur(bgr, 3)
    residual = np.abs(bgr.astype(np.float32) - median.astype(np.float32))
    block_std = split_blocks(gray, DCT_SIZE).std(axis=(1, 2))
    low, middle, high = np.percentile(block_std, [5, 50, 95])
    features = {
        "noise_immerkaer_gray": immerkaer_sigma(gray),
        "noise_median_residual_mean": float(residual.mean()),
        "noise_median_residual_p99": float(np.percentile(residual, 99)),
        "noise_impulse_share": float((residual > IMPULSE_THRESHOLD).mean()),
        "noise_block_std_p05": float(low),
        "noise_block_std_p50": float(middle),
        "noise_block_std_p95": float(high),
    }
    for index, name in enumerate(("blue", "green", "red")):
        features[f"noise_immerkaer_{name}"] = immerkaer_sigma(bgr[:, :, index].astype(np.float32))
    return features


def blockiness(gray, period, axis):
    """Отношение перепадов на границах блоков к перепадам внутри блоков."""
    profile = np.abs(np.diff(gray, axis=axis)).mean(axis=1 - axis)
    on_boundary = (np.arange(profile.size) + 1) % period == 0
    return float(profile[on_boundary].mean() / (profile[~on_boundary].mean() + EPS))


def compression_features(gray):
    """Блочность на нескольких периодах и распределение коэффициентов DCT."""
    features = {}
    for period in BLOCK_PERIODS:
        features[f"comp_block{period}_horizontal"] = blockiness(gray, period, 1)
        features[f"comp_block{period}_vertical"] = blockiness(gray, period, 0)

    coefficients = np.abs(dctn(split_blocks(gray, DCT_SIZE), axes=(1, 2), norm="ortho"))
    rows, columns = np.indices((DCT_SIZE, DCT_SIZE))
    frequency = rows + columns
    bands = {
        "low": (frequency > 0) & (frequency <= 4),
        "mid": (frequency > 4) & (frequency <= 9),
        "high": frequency > 9,
    }
    for name, mask in bands.items():
        selected = coefficients[:, mask]
        features[f"comp_dct_{name}_mean"] = float(selected.mean())
        features[f"comp_dct_{name}_zero_share"] = float((selected < DCT_ZERO_THRESHOLD).mean())
    return features


def spectral_features(gray):
    """Распределение энергии спектра по частотным кольцам и наклон спектра."""
    spectrum = np.abs(np.fft.rfft2(gray - gray.mean())) ** 2
    frequency_y = np.fft.fftfreq(gray.shape[0])[:, None]
    frequency_x = np.fft.rfftfreq(gray.shape[1])[None, :]
    radius = np.minimum(np.hypot(frequency_y, frequency_x) / 0.5, 1.0)
    bins = np.minimum((radius * RADIAL_BINS).astype(int), RADIAL_BINS - 1)
    energy = np.bincount(bins.ravel(), weights=spectrum.ravel(), minlength=RADIAL_BINS)
    counts = np.bincount(bins.ravel(), minlength=RADIAL_BINS)
    shares = energy / (energy.sum() + EPS)
    centers = (np.arange(RADIAL_BINS) + 0.5) / RADIAL_BINS
    slope = np.polyfit(np.log(centers), np.log(energy / counts + EPS), 1)[0]
    features = {f"spectrum_ring{index}_share": float(share) for index, share in enumerate(shares)}
    features["spectrum_slope"] = float(slope)
    return features


def ggd_shape(values):
    """Оценка параметра формы обобщённого нормального распределения методом моментов."""
    ratio = (values**2).mean() / (np.abs(values).mean() ** 2 + EPS)
    return float(GGD_SHAPES[np.argmin(np.abs(GGD_RATIOS - ratio))])


def nss_features(gray):
    """Статистики нормализованных яркостей (MSCN) в духе модели BRISQUE."""
    features = {}
    for scale in NSS_SCALES:
        size = (gray.shape[1] // scale, gray.shape[0] // scale)
        image = gray if scale == 1 else cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        mean = cv2.GaussianBlur(image, (7, 7), 7 / 6)
        variance = cv2.GaussianBlur(image**2, (7, 7), 7 / 6) - mean**2
        mscn = (image - mean) / (np.sqrt(np.abs(variance)) + 1)
        horizontal = mscn[:, :-1] * mscn[:, 1:]
        vertical = mscn[:-1, :] * mscn[1:, :]
        is_constant = mscn.std() < EPS
        prefix = f"nss_s{scale}"
        features.update(
            {
                f"{prefix}_shape": ggd_shape(mscn),
                f"{prefix}_var": float(mscn.var()),
                f"{prefix}_kurt": 0.0 if is_constant else float(kurtosis(mscn.ravel())),
                f"{prefix}_skew": 0.0 if is_constant else float(skew(mscn.ravel())),
                f"{prefix}_pair_horizontal_mean": float(horizontal.mean()),
                f"{prefix}_pair_horizontal_var": float(horizontal.var()),
                f"{prefix}_pair_vertical_mean": float(vertical.mean()),
                f"{prefix}_pair_vertical_var": float(vertical.var()),
            }
        )
    return features


def texture_features(gray):
    """Признаки GLCM, гистограмма LBP и энергия откликов фильтров Габора."""
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    quantized = (gray_u8 // (256 // GLCM_LEVELS)).astype(np.uint8)
    matrix = graycomatrix(
        quantized, GLCM_DISTANCES, GLCM_ANGLES, levels=GLCM_LEVELS, symmetric=True, normed=True
    )
    features = {}
    for prop in GLCM_PROPERTIES:
        values = graycoprops(matrix, prop).mean(axis=1)
        for distance, value in zip(GLCM_DISTANCES, values):
            features[f"texture_glcm_{prop}_d{distance}"] = float(value)

    pattern = local_binary_pattern(gray_u8, LBP_POINTS, LBP_RADIUS, method="uniform")
    histogram = np.bincount(pattern.astype(int).ravel(), minlength=LBP_POINTS + 2) / pattern.size
    for index, share in enumerate(histogram):
        features[f"texture_lbp_{index}"] = float(share)

    for (wavelength, orientation), kernel in GABOR_BANK.items():
        response = cv2.filter2D(gray, cv2.CV_32F, kernel)
        features[f"texture_gabor_w{wavelength}_o{orientation}"] = float(np.abs(response).mean())
    return features


def image_features(bgr):
    """Считает все признаки для одного изображения в формате BGR."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return {
        **color_features(bgr),
        **sharpness_features(gray),
        **noise_features(bgr, gray),
        **compression_features(gray),
        **spectral_features(gray),
        **nss_features(gray),
        **texture_features(gray),
    }


def process_image(path_text):
    """Читает файл и считает его признаки.

    Файл читается через numpy, поскольку cv2.imread в Windows не открывает пути с не-ASCII символами.
    """
    bgr = cv2.imdecode(np.fromfile(path_text, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Не удалось прочитать изображение: {path_text}")
    return image_features(bgr)


def find_images(names, images_dir):
    """Возвращает пути к изображениям в порядке имён, проверяя наличие всех файлов."""
    images = {path.name: path for path in Path(images_dir).rglob("*.png")}
    missing = [name for name in names if name not in images]
    if missing:
        raise ValueError(f"В папке {images_dir} нет {len(missing)} изображений, например: {missing[:5]}")
    return [str(images[name]) for name in names]


def compute_rows(paths):
    """Считает признаки параллельно и сообщает о ходе работы.

    Процессы создаются через spawn: после использования OpenCV в основном процессе пул на основе fork
    блокируется. Таймаут превращает возможное зависание в ошибку. Ограничение потоков наследуется
    дочерними процессами и снимается после расчёта, чтобы не замедлять обучение моделей.
    """
    rows = []
    started = time.perf_counter()
    previous = {name: os.environ.get(name) for name in THREAD_LIMITS}
    os.environ.update(THREAD_LIMITS)
    try:
        with multiprocessing.get_context("spawn").Pool(os.cpu_count()) as pool:
            results = pool.imap(process_image, paths)
            for count in range(1, len(paths) + 1):
                rows.append(results.next(timeout=IMAGE_TIMEOUT_SECONDS))
                if count % PROGRESS_STEP == 0:
                    elapsed = time.perf_counter() - started
                    print(f"{count}/{len(paths)} изображений, {elapsed:.0f} с", flush=True)
    finally:
        for name, value in previous.items():
            os.environ.pop(name, None)
            if value is not None:
                os.environ[name] = value
    return rows


def extract_features(names, images_dir, id_column):
    """Строит таблицу признаков для изображений с заданными именами."""
    names = list(names)
    table = pd.DataFrame(compute_rows(find_images(names, images_dir)))
    table.insert(0, id_column, names)
    return table

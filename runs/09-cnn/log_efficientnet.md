# Лог прогона 09. CNN: EfficientNetB0 на целых изображениях

Пояснения к прогону — в [README.md](README.md). Время в начале строки отсчитывается от запуска ядра.

```text
      3.4s  bacbf683916abf880d9f4a9f9d86670fdad8bf0a Сравнение EfficientNet на целых изображениях и на вырезках
      3.4s  
     25.4s  ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
     25.4s  bigframes 2.51.0 requires google-cloud-bigquery-storage<3.0.0,>=2.30.0, which is not installed.
     25.4s  ydata-profiling 4.18.4 requires matplotlib<=3.10,>=3.5, but you have matplotlib 3.10.9 which is incompatible.
     25.7s  {"python": "3.13.16", "gpu": "Tesla T4\nTesla T4", "tensorflow": "2.21.0", "keras": "3.13.2", "numpy": "2.1.3", "pandas": "2.3.3", "scikit-learn": "1.6.1", "scipy": "1.16.3", "scikit-image": "0.25.2", "opencv-python-headless": "4.14.0.94", "matplotlib": "3.10.9", "pyyaml": "6.0.3", "pytest": "8.4.2", "joblib": "1.6.0", "lightgbm": "4.6.0", "xgboost": "3.4.1", "catboost": "1.2.10", "optuna": "5.0.0"}
    114.4s  .....................................................................    [100%]
    114.4s  69 passed in 83.81s (0:01:23)
    120.6s  == run_cnn.yaml
    131.0s  Downloading data from https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5
    132.2s  16705208/16705208 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step
  23828.2s  cnn_efficientnet: MAE 0.4930
  23828.3s             model      mae  fold_std  train_mae    mae_1    mae_2    mae_3    mae_4    mae_5
  23828.3s  cnn_efficientnet 0.492996  0.020807   0.330481 0.641302 0.106812 0.375661 0.726141 0.615062
  23838.5s  /usr/local/lib/python3.13/dist-packages/mistune.py:435: SyntaxWarning: invalid escape sequence '\|'
  23838.5s    cells[i][c] = re.sub('\\\\\|', '|', cell)
  23838.7s  /usr/local/lib/python3.13/dist-packages/nbconvert/filters/filter_links.py:36: SyntaxWarning: invalid escape sequence '\_'
  23838.7s    text = re.sub(r'_', '\_', text) # Escape underscores in display text
  23839.5s  [NbConvertApp] Converting notebook __script__.ipynb to html
  23840.0s  [NbConvertApp] Writing 287996 bytes to __results__.html
```

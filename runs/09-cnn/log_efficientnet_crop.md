# Лог прогона 09. CNN: EfficientNetB0 на вырезках

Пояснения к прогону — в [README.md](README.md). Время в начале строки отсчитывается от запуска ядра.

```text
      2.0s  bacbf683916abf880d9f4a9f9d86670fdad8bf0a Сравнение EfficientNet на целых изображениях и на вырезках
      2.0s  
     20.2s  ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
     20.2s  bigframes 2.51.0 requires google-cloud-bigquery-storage<3.0.0,>=2.30.0, which is not installed.
     20.2s  ydata-profiling 4.18.4 requires matplotlib<=3.10,>=3.5, but you have matplotlib 3.10.9 which is incompatible.
     20.7s  {"python": "3.13.16", "gpu": "Tesla T4\nTesla T4", "tensorflow": "2.21.0", "keras": "3.13.2", "numpy": "2.1.3", "pandas": "2.3.3", "scikit-learn": "1.6.1", "scipy": "1.16.3", "scikit-image": "0.25.2", "opencv-python-headless": "4.14.0.94", "matplotlib": "3.10.9", "pyyaml": "6.0.3", "pytest": "8.4.2", "joblib": "1.6.0", "lightgbm": "4.6.0", "xgboost": "3.4.1", "catboost": "1.2.10", "optuna": "5.0.0"}
    123.9s  .....................................................................    [100%]
    123.9s  69 passed in 98.20s (0:01:38)
    130.0s  == run_cnn.yaml
    143.1s  Downloading data from https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5
    143.3s  16705208/16705208 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step
   8984.1s  cnn_efficientnet_crop: MAE 0.5783
   8984.2s                  model      mae  fold_std  train_mae    mae_1    mae_2    mae_3    mae_4    mae_5
   8984.2s  cnn_efficientnet_crop 0.578265  0.012507   0.510015 0.798109 0.113833 0.435312 0.723967 0.820102
   8994.3s  /usr/local/lib/python3.13/dist-packages/mistune.py:435: SyntaxWarning: invalid escape sequence '\|'
   8994.3s    cells[i][c] = re.sub('\\\\\|', '|', cell)
   8994.5s  /usr/local/lib/python3.13/dist-packages/nbconvert/filters/filter_links.py:36: SyntaxWarning: invalid escape sequence '\_'
   8994.5s    text = re.sub(r'_', '\_', text) # Escape underscores in display text
   8995.3s  [NbConvertApp] Converting notebook __script__.ipynb to html
   8995.8s  [NbConvertApp] Writing 288001 bytes to __results__.html
```

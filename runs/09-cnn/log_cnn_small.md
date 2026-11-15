# Лог прогона 09. CNN: сеть с нуля на вырезках

Пояснения к прогону — в [README.md](README.md). Время в начале строки отсчитывается от запуска ядра.

```text
      2.1s  13ebecd7e6edf34a1b48139645bff2c1ade653d8 Обучение CNN на вырезках со смешанной точностью
      2.1s  
     24.9s  ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
     24.9s  bigframes 2.51.0 requires google-cloud-bigquery-storage<3.0.0,>=2.30.0, which is not installed.
     24.9s  ydata-profiling 4.18.4 requires matplotlib<=3.10,>=3.5, but you have matplotlib 3.10.9 which is incompatible.
     25.2s  {"python": "3.13.16", "gpu": "Tesla T4\nTesla T4", "tensorflow": "2.21.0", "keras": "3.13.2", "numpy": "2.1.3", "pandas": "2.3.3", "scikit-learn": "1.6.1", "scipy": "1.16.3", "scikit-image": "0.25.2", "opencv-python-headless": "4.14.0.94", "matplotlib": "3.10.9", "pyyaml": "6.0.3", "pytest": "8.4.2", "joblib": "1.6.0", "lightgbm": "4.6.0", "xgboost": "3.4.1", "catboost": "1.2.10", "optuna": "5.0.0"}
    120.0s  ....................................................................     [100%]
    120.0s  68 passed in 89.89s (0:01:29)
    126.5s  == run_cnn.yaml
   9759.9s  cnn_small: MAE 0.8112
  12032.6s      model     mae  fold_std  train_mae    mae_1   mae_2    mae_3    mae_4   mae_5
  12032.6s  cnn_small 0.81124  0.012652   0.800125 1.239186 0.11044 0.527441 0.793112 1.38602
  12037.0s  /usr/local/lib/python3.13/dist-packages/mistune.py:435: SyntaxWarning: invalid escape sequence '\|'
  12037.0s    cells[i][c] = re.sub('\\\\\|', '|', cell)
  12037.3s  /usr/local/lib/python3.13/dist-packages/nbconvert/filters/filter_links.py:36: SyntaxWarning: invalid escape sequence '\_'
  12037.3s    text = re.sub(r'_', '\_', text) # Escape underscores in display text
  12038.1s  [NbConvertApp] Converting notebook __script__.ipynb to html
  12038.7s  [NbConvertApp] Writing 287989 bytes to __results__.html
```

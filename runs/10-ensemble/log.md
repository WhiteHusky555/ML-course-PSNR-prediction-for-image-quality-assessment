# Лог прогона 10. Ансамбль

Пояснения к прогону — в [README.md](README.md). Время в начале строки отсчитывается от запуска ядра.

```text
      3.2s  94f1df6635046276e01250f97ce950ee0da2b023 fix: префикс прогнозов без двоеточия для стекинга на LightGBM
      3.2s  
     45.1s  ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
     45.1s  bigframes 2.51.0 requires google-cloud-bigquery-storage<3.0.0,>=2.30.0, which is not installed.
     45.1s  ydata-profiling 4.18.4 requires matplotlib<=3.10,>=3.5, but you have matplotlib 3.10.9 which is incompatible.
     45.6s  {"python": "3.13.18", "tensorflow": "2.21.0", "keras": "3.13.2", "numpy": "2.1.3", "pandas": "2.3.3", "scikit-learn": "1.6.1", "scipy": "1.16.3", "lightgbm": "4.6.0", "xgboost": "3.4.1", "catboost": "1.2.10"}
    143.4s  ........................................................................ [ 88%]
    146.0s  .........                                                                [100%]
    146.0s  81 passed in 92.94s (0:01:32)
    149.4s  == run_ensemble_base.yaml
    287.3s  lightgbm: MAE 0.6490
    666.4s  xgboost: MAE 0.6456
    876.7s  catboost: MAE 0.6596
    877.1s  WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    877.1s  I0000 00:00:1794814531.424371    2501 cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
    880.0s  WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    880.0s  I0000 00:00:1794814534.291249    2501 cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
    880.6s  E0000 00:00:1794814534.878070    2501 cuda_platform.cc:52] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)
   1038.0s  mlp_silu: MAE 0.6518
   1188.7s  mlp_elu: MAE 0.6523
   1188.9s     model      mae  fold_std  train_mae    mae_1    mae_2    mae_3    mae_4    mae_5
   1188.9s   xgboost 0.645552  0.008294   0.172799 0.965504 0.090532 0.382378 0.700832 1.088514
   1188.9s  lightgbm 0.648960  0.011838   0.315473 0.976508 0.085915 0.401652 0.683706 1.097021
   1188.9s  mlp_silu 0.651777  0.009660   0.532700 1.011382 0.103112 0.366338 0.696819 1.081235
   1188.9s   mlp_elu 0.652346  0.007386   0.562419 0.997663 0.105639 0.380897 0.713000 1.064530
   1188.9s  catboost 0.659616  0.011548   0.420543 0.993994 0.090508 0.418940 0.705013 1.089626
   1191.1s  == run_ensemble.yaml
   1196.3s  mean: MAE 0.5746
   1223.0s  weights: MAE 0.4855
   1230.0s  stacking_predictions: MAE 0.4606
   1255.1s  stacking_features: MAE 0.4625
   1255.3s                 model      mae  fold_std  train_mae    mae_1    mae_2    mae_3    mae_4    mae_5
   1255.3s  stacking_predictions 0.460591  0.018934   0.424417 0.629392 0.087419 0.340971 0.647127 0.598048
   1255.3s     stacking_features 0.462453  0.017721   0.402444 0.625238 0.086671 0.349237 0.651674 0.599446
   1255.3s               weights 0.485548  0.018896   0.485293 0.634906 0.102266 0.370680 0.703860 0.616027
   1255.3s                  mean 0.574569  0.011170   0.574569 0.838606 0.091624 0.372231 0.666888 0.903497
   1258.7s  /usr/local/lib/python3.13/dist-packages/mistune.py:435: SyntaxWarning: invalid escape sequence '\|'
   1258.7s    cells[i][c] = re.sub('\\\\\|', '|', cell)
   1259.1s  /usr/local/lib/python3.13/dist-packages/nbconvert/filters/filter_links.py:36: SyntaxWarning: invalid escape sequence '\_'
   1259.1s    text = re.sub(r'_', '\_', text) # Escape underscores in display text
   1260.2s  [NbConvertApp] Converting notebook __script__.ipynb to html
   1260.9s  [NbConvertApp] Writing 285205 bytes to __results__.html
```

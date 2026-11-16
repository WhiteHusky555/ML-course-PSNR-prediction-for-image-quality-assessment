# Лог прогона 11. Признаки следов искажений: пересчёт таблицы, базовые модели и ансамбль

Пояснения к прогону — в [README.md](README.md). Время в начале строки отсчитывается от запуска ядра.

```text
      2.5s  1e4a44f428d8ba5861c2db537d06156728ccd6c4 feat: признаки следов искажений: гистограмма, шум, изменение размера
      2.5s  
     32.3s  ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
     32.3s  bigframes 2.51.0 requires google-cloud-bigquery-storage<3.0.0,>=2.30.0, which is not installed.
     32.3s  ydata-profiling 4.18.4 requires matplotlib<=3.10,>=3.5, but you have matplotlib 3.10.9 which is incompatible.
     32.7s  {"python": "3.13.18", "tensorflow": "2.21.0", "keras": "3.13.2", "numpy": "2.1.3", "pandas": "2.3.3", "scikit-learn": "1.6.1", "scipy": "1.16.3", "lightgbm": "4.6.0", "xgboost": "3.4.1", "catboost": "1.2.10"}
    117.8s  ........................................................................ [ 85%]
    124.1s  ............                                                             [100%]
    124.1s  84 passed in 84.70s (0:01:24)
    127.2s  == run_ensemble_base.yaml
    310.9s  1000/16520 изображений, 161 с
    464.4s  2000/16520 изображений, 315 с
    613.7s  3000/16520 изображений, 464 с
    763.4s  4000/16520 изображений, 614 с
    913.5s  5000/16520 изображений, 764 с
   1062.2s  6000/16520 изображений, 913 с
   1218.9s  7000/16520 изображений, 1069 с
   1387.4s  8000/16520 изображений, 1238 с
   1562.2s  9000/16520 изображений, 1413 с
   1719.9s  10000/16520 изображений, 1570 с
   1889.2s  11000/16520 изображений, 1740 с
   2042.2s  12000/16520 изображений, 1893 с
   2192.8s  13000/16520 изображений, 2043 с
   2341.0s  14000/16520 изображений, 2191 с
   2492.6s  15000/16520 изображений, 2343 с
   2655.8s  16000/16520 изображений, 2506 с
   2890.2s  lightgbm: MAE 0.6363
   3288.0s  xgboost: MAE 0.6370
   3527.7s  catboost: MAE 0.6477
   3530.3s  WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
   3530.3s  I0000 00:00:1794833314.314581    2501 cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
   3540.3s  WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
   3540.3s  I0000 00:00:1794833324.256772    2501 cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
   3541.5s  E0000 00:00:1794833325.497336    2501 cuda_platform.cc:52] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)
   3703.4s  mlp_silu: MAE 0.6378
   3853.7s  mlp_elu: MAE 0.6360
   3854.0s     model      mae  fold_std  train_mae    mae_1    mae_2    mae_3    mae_4    mae_5
   3854.0s   mlp_elu 0.635972  0.006660   0.542474 0.996618 0.102043 0.366013 0.683047 1.032140
   3854.0s  lightgbm 0.636296  0.009981   0.298686 0.978516 0.086725 0.376711 0.656373 1.083153
   3854.0s   xgboost 0.636996  0.008257   0.155806 0.970525 0.090647 0.367205 0.671197 1.085408
   3854.0s  mlp_silu 0.637831  0.008657   0.510734 1.009875 0.101494 0.354249 0.674250 1.049289
   3854.0s  catboost 0.647715  0.008488   0.401622 0.997508 0.091337 0.396923 0.680061 1.072746
   3856.3s  == run_ensemble.yaml
   3863.3s  mean: MAE 0.5647
   3890.2s  weights: MAE 0.4846
   3897.3s  stacking_predictions: MAE 0.4508
   3924.2s  stacking_features: MAE 0.4538
   3924.4s                 model      mae  fold_std  train_mae    mae_1    mae_2    mae_3    mae_4    mae_5
   3924.4s  stacking_predictions 0.450787  0.018095   0.415084 0.632679 0.088016 0.323344 0.613802 0.596094
   3924.4s     stacking_features 0.453783  0.017061   0.392352 0.625658 0.087085 0.334326 0.626391 0.595451
   3924.4s               weights 0.484607  0.018437   0.484234 0.636143 0.101852 0.368224 0.698901 0.617914
   3924.4s                  mean 0.564663  0.010522   0.564664 0.839359 0.090996 0.357907 0.647318 0.887738
   3928.0s  /usr/local/lib/python3.13/dist-packages/mistune.py:435: SyntaxWarning: invalid escape sequence '\|'
   3928.0s    cells[i][c] = re.sub('\\\\\|', '|', cell)
   3928.4s  /usr/local/lib/python3.13/dist-packages/nbconvert/filters/filter_links.py:36: SyntaxWarning: invalid escape sequence '\_'
   3928.4s    text = re.sub(r'_', '\_', text) # Escape underscores in display text
   3929.5s  [NbConvertApp] Converting notebook __script__.ipynb to html
   3930.2s  [NbConvertApp] Writing 286168 bytes to __results__.html
```

# Лог прогона 09. CNN: пробный запуск с одной эпохой

Пояснения к прогону — в [README.md](README.md). Время в начале строки отсчитывается от запуска ядра.

```text
      1.9s  60f74889652a46fe38f7f90c18eb3d3f08fd05c0 Исправлен расход памяти при чтении изображений в CNN
      1.9s  
     24.1s  ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
     24.1s  bigframes 2.51.0 requires google-cloud-bigquery-storage<3.0.0,>=2.30.0, which is not installed.
     24.1s  ydata-profiling 4.18.4 requires matplotlib<=3.10,>=3.5, but you have matplotlib 3.10.9 which is incompatible.
     24.4s  {"python": "3.13.16", "tensorflow": "2.21.0", "keras": "3.13.2", "numpy": "2.1.3", "pandas": "2.3.3", "scikit-learn": "1.6.1", "scipy": "1.16.3", "scikit-image": "0.25.2", "opencv-python-headless": "4.14.0.94", "matplotlib": "3.10.9", "pyyaml": "6.0.3", "pytest": "8.4.2", "joblib": "1.6.0", "lightgbm": "4.6.0", "xgboost": "3.4.1", "catboost": "1.2.10", "optuna": "5.0.0"}
    107.1s  .................................................................        [100%]
    107.1s  65 passed in 77.59s (0:01:17)
    113.4s  == run_cnn.yaml
   4429.1s  cnn_small: MAE 0.8919
   4430.1s  Downloading data from https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5
   4430.2s  16705208/16705208 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step
   4470.5s  WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
   4470.5s  E0000 00:00:1794561598.942724    1766 meta_optimizer.cc:967] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape insequential_5_1/efficientnetb0_1/block2b_drop_1/stateless_dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
   5134.5s  E0000 00:00:1794562262.962833    1766 meta_optimizer.cc:967] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape insequential_5_1/efficientnetb0_1/block2b_drop_1/stateless_dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
   5698.0s  E0000 00:00:1794562826.396009    1766 meta_optimizer.cc:967] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape insequential_6_1/efficientnetb0_1/block2b_drop_1/stateless_dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
   6365.4s  E0000 00:00:1794563493.835021    1766 meta_optimizer.cc:967] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape insequential_6_1/efficientnetb0_1/block2b_drop_1/stateless_dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
   6934.6s  E0000 00:00:1794564062.996835    1766 meta_optimizer.cc:967] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape insequential_7_1/efficientnetb0_1/block2b_drop_1/stateless_dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
   7597.8s  E0000 00:00:1794564726.225822    1766 meta_optimizer.cc:967] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape insequential_7_1/efficientnetb0_1/block2b_drop_1/stateless_dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
   8181.8s  E0000 00:00:1794565310.254486    1766 meta_optimizer.cc:967] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape insequential_8_1/efficientnetb0_1/block2b_drop_1/stateless_dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
   8848.1s  E0000 00:00:1794565976.516356    1766 meta_optimizer.cc:967] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape insequential_8_1/efficientnetb0_1/block2b_drop_1/stateless_dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
   9430.2s  E0000 00:00:1794566558.594891    1766 meta_optimizer.cc:967] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape insequential_9_1/efficientnetb0_1/block2b_drop_1/stateless_dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
  10093.8s  E0000 00:00:1794567222.230627    1766 meta_optimizer.cc:967] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape insequential_9_1/efficientnetb0_1/block2b_drop_1/stateless_dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
  10631.7s  cnn_efficientnet: MAE 0.6782
  10677.8s  E0000 00:00:1794567806.193215    1766 meta_optimizer.cc:967] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape insequential_10_1/efficientnetb0_1/block2b_drop_1/stateless_dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
  11502.7s  E0000 00:00:1794568631.157682    1766 meta_optimizer.cc:967] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape insequential_10_1/efficientnetb0_1/block2b_drop_1/stateless_dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
  11649.0s             model      mae  fold_std  train_mae    mae_1    mae_2    mae_3    mae_4    mae_5
  11649.0s  cnn_efficientnet 0.678162  0.018318   0.625248 0.968271 0.143789 0.515862 0.787366 0.975521
  11649.0s         cnn_small 0.891936  0.021497   0.888486 1.339709 0.129784 0.527336 0.942002 1.520850
  11666.4s  /usr/local/lib/python3.13/dist-packages/mistune.py:435: SyntaxWarning: invalid escape sequence '\|'
  11666.4s    cells[i][c] = re.sub('\\\\\|', '|', cell)
  11666.6s  /usr/local/lib/python3.13/dist-packages/nbconvert/filters/filter_links.py:36: SyntaxWarning: invalid escape sequence '\_'
  11666.6s    text = re.sub(r'_', '\_', text) # Escape underscores in display text
  11667.5s  [NbConvertApp] Converting notebook __script__.ipynb to html
  11668.0s  [NbConvertApp] Writing 285963 bytes to __results__.html
```

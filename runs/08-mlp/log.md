# Лог прогона 08. Полносвязная нейросеть (MLP)

Пояснения к прогону — в [README.md](README.md). Время в начале строки отсчитывается от запуска ядра.

```text
      2.5s  bd8538b2eb0167afe2315e3b8268754e510c3ae5 Добавлены сочетания параметров MLP, подобранный LightGBM внесён в таблицу результатов
      2.5s  
     34.9s  ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
     34.9s  bigframes 2.51.0 requires google-cloud-bigquery-storage<3.0.0,>=2.30.0, which is not installed.
     34.9s  ydata-profiling 4.18.4 requires matplotlib<=3.10,>=3.5, but you have matplotlib 3.10.9 which is incompatible.
     35.3s  {"python": "3.13.18", "tensorflow": "2.21.0", "keras": "3.13.2", "numpy": "2.1.3", "pandas": "2.3.3", "scikit-learn": "1.6.1", "scipy": "1.16.3", "scikit-image": "0.25.2", "opencv-python-headless": "4.14.0.94", "matplotlib": "3.10.9", "pyyaml": "6.0.3", "pytest": "8.4.2", "joblib": "1.6.0", "lightgbm": "4.6.0", "xgboost": "3.4.1", "catboost": "1.2.10", "optuna": "5.0.0"}
     92.3s  ............................................................             [100%]
     92.3s  60 passed in 52.50s
     95.0s  == run_mlp.yaml
    100.1s  WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    100.1s  I0000 00:00:1794464922.777876    1074 cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
    103.1s  WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    103.1s  I0000 00:00:1794464925.719172    1074 cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
    103.6s  E0000 00:00:1794464926.274851    1074 cuda_platform.cc:52] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)
    153.7s  mlp_simple: MAE 0.7087
    233.4s  mlp_depth hidden_units=[256, 128]: MAE 0.7343
    326.5s  mlp_depth hidden_units=[256, 128, 64]: MAE 0.7297
    483.4s  mlp_depth hidden_units=[512, 256, 128, 64]: MAE 0.7237
    600.7s  mlp_batchnorm: MAE 0.7389
    734.8s  mlp_dropout dropout=0.1: MAE 0.6803
    870.3s  mlp_dropout dropout=0.2: MAE 0.6626
   1007.4s  mlp_dropout dropout=0.3: MAE 0.6675
   1143.8s  mlp_dropout dropout=0.5: MAE 0.6744
   1279.7s  mlp_base: MAE 0.6658
   1362.6s  mlp_width hidden_units=[64, 32, 16]: MAE 0.6702
   1588.6s  mlp_width hidden_units=[512, 256, 128]: MAE 0.6666
   2076.8s  mlp_width hidden_units=[1024, 512, 256]: MAE 0.6751
   2233.8s  mlp_activation activation=gelu: MAE 0.6641
   2380.6s  mlp_activation activation=silu: MAE 0.6603
   2518.3s  mlp_activation activation=tanh: MAE 0.6652
   2658.9s  mlp_activation activation=elu: MAE 0.6523
   2797.9s  mlp_optimizer optimizer=adamw: MAE 0.6684
   2928.5s  mlp_optimizer optimizer=rmsprop: MAE 0.6658
   3058.0s  mlp_sgd learning_rate=0.01: MAE 0.6746
   3182.7s  mlp_sgd learning_rate=0.05: MAE 0.6630
   3319.1s  mlp_learning_rate learning_rate=0.0003: MAE 0.6751
   3454.7s  mlp_learning_rate learning_rate=0.003: MAE 0.6691
   3589.3s  mlp_learning_rate learning_rate=0.01: MAE 0.6642
   3882.8s  mlp_batch_size batch_size=64: MAE 0.6672
   3975.6s  mlp_batch_size batch_size=1024: MAE 0.6748
   4049.9s  mlp_epochs epochs=50: MAE 0.6678
   4308.2s  mlp_epochs epochs=200: MAE 0.6752
   4442.4s  mlp_loss loss=mse: MAE 0.7267
   4575.0s  mlp_loss loss=huber: MAE 0.6961
   4714.5s  mlp_combined activation=elu dropout=0.2: MAE 0.6595
   4853.6s  mlp_combined activation=elu dropout=0.3: MAE 0.6714
   5002.6s  mlp_combined activation=silu dropout=0.2: MAE 0.6518
   5153.7s  mlp_combined activation=silu dropout=0.3: MAE 0.6561
   5286.2s  mlp_combined_sgd: MAE 0.6713
   5322.1s                                       model      mae  fold_std  train_mae    mae_1    mae_2    mae_3    mae_4    mae_5
   5322.1s    mlp_combined activation=silu dropout=0.2 0.651777  0.009660   0.532700 1.011382 0.103112 0.366338 0.696819 1.081235
   5322.1s               mlp_activation activation=elu 0.652346  0.007386   0.562419 0.997663 0.105639 0.380897 0.713000 1.064530
   5322.1s    mlp_combined activation=silu dropout=0.3 0.656111  0.008257   0.565573 1.013808 0.104737 0.380676 0.700832 1.080503
   5322.1s     mlp_combined activation=elu dropout=0.2 0.659510  0.007376   0.599085 1.007841 0.108031 0.401446 0.709469 1.070762
   5322.1s              mlp_activation activation=silu 0.660273  0.010448   0.467297 1.025306 0.100548 0.363751 0.713528 1.098233
   5322.1s                     mlp_dropout dropout=0.2 0.662558  0.010726   0.483805 1.034063 0.105445 0.370435 0.696432 1.106415
   5322.1s                  mlp_sgd learning_rate=0.05 0.662960  0.007826   0.430633 1.026854 0.104201 0.375958 0.696077 1.111708
   5322.1s              mlp_activation activation=gelu 0.664122  0.008862   0.414374 1.029776 0.099091 0.367775 0.715260 1.108711
   5322.1s        mlp_learning_rate learning_rate=0.01 0.664188  0.006500   0.363157 1.035506 0.098492 0.357135 0.714790 1.115018
   5322.1s              mlp_activation activation=tanh 0.665204  0.011198   0.521435 1.031977 0.109582 0.389397 0.699605 1.095458
   5322.1s             mlp_optimizer optimizer=rmsprop 0.665765  0.008031   0.415742 1.041120 0.100602 0.369319 0.700727 1.117058
   5322.1s                                    mlp_base 0.665826  0.009213   0.419001 1.034437 0.100392 0.370113 0.705094 1.119093
   5322.1s      mlp_width hidden_units=[512, 256, 128] 0.666616  0.007900   0.335732 1.034445 0.097845 0.361631 0.718707 1.120452
   5322.1s                mlp_batch_size batch_size=64 0.667238  0.009375   0.412476 1.045458 0.100003 0.368696 0.702237 1.119798
   5322.1s                     mlp_dropout dropout=0.3 0.667471  0.011147   0.531069 1.042070 0.105269 0.382131 0.686212 1.121675
   5322.1s                        mlp_epochs epochs=50 0.667800  0.010951   0.538205 1.033288 0.105078 0.399389 0.700001 1.101246
   5322.1s               mlp_optimizer optimizer=adamw 0.668358  0.006080   0.421575 1.042157 0.100868 0.371135 0.707336 1.120295
   5322.1s       mlp_learning_rate learning_rate=0.003 0.669106  0.008431   0.373240 1.047204 0.099093 0.361275 0.711740 1.126218
   5322.1s         mlp_width hidden_units=[64, 32, 16] 0.670202  0.008789   0.569271 1.032497 0.108341 0.399395 0.697989 1.112789
   5322.1s                            mlp_combined_sgd 0.671259  0.008009   0.625791 1.020202 0.113868 0.418346 0.717495 1.086385
   5322.1s     mlp_combined activation=elu dropout=0.3 0.671372  0.006691   0.629299 1.023880 0.112011 0.421169 0.712526 1.087274
   5322.1s                     mlp_dropout dropout=0.5 0.674383  0.007664   0.607279 1.032385 0.113209 0.412725 0.697568 1.116028
   5322.1s                  mlp_sgd learning_rate=0.01 0.674550  0.009932   0.549293 1.041336 0.107408 0.409306 0.702644 1.112058
   5322.1s              mlp_batch_size batch_size=1024 0.674838  0.011086   0.520706 1.039461 0.105219 0.397847 0.713810 1.117852
   5322.1s      mlp_learning_rate learning_rate=0.0003 0.675084  0.009867   0.553552 1.038412 0.106115 0.415054 0.709315 1.106523
   5322.1s     mlp_width hidden_units=[1024, 512, 256] 0.675091  0.005933   0.278550 1.040088 0.096163 0.359454 0.732491 1.147260
   5322.1s                       mlp_epochs epochs=200 0.675236  0.004852   0.286505 1.040838 0.097408 0.365020 0.722922 1.149992
   5322.1s                     mlp_dropout dropout=0.1 0.680345  0.009238   0.428241 1.061289 0.104445 0.375523 0.711233 1.149237
   5322.1s                         mlp_loss loss=huber 0.696136  0.003320   0.373984 1.068845 0.127761 0.382785 0.746862 1.154425
   5322.1s                                  mlp_simple 0.708664  0.030296   0.581004 1.019560 0.156943 0.479436 0.809982 1.077398
   5322.1s  mlp_depth hidden_units=[512, 256, 128, 64] 0.723732  0.011082   0.158055 1.136165 0.099761 0.386442 0.787296 1.208995
   5322.1s                           mlp_loss loss=mse 0.726684  0.012878   0.395843 1.077427 0.143830 0.412270 0.807097 1.192796
   5322.1s       mlp_depth hidden_units=[256, 128, 64] 0.729742  0.010987   0.228806 1.127616 0.100040 0.386757 0.797852 1.236446
   5322.1s           mlp_depth hidden_units=[256, 128] 0.734323  0.011767   0.302940 1.112485 0.106653 0.379528 0.858971 1.213976
   5322.1s                               mlp_batchnorm 0.738874  0.012974   0.326253 1.149116 0.117649 0.419494 0.804249 1.203861
   5339.5s  /usr/local/lib/python3.13/dist-packages/mistune.py:435: SyntaxWarning: invalid escape sequence '\|'
   5339.5s    cells[i][c] = re.sub('\\\\\|', '|', cell)
   5339.9s  /usr/local/lib/python3.13/dist-packages/nbconvert/filters/filter_links.py:36: SyntaxWarning: invalid escape sequence '\_'
   5339.9s    text = re.sub(r'_', '\_', text) # Escape underscores in display text
   5340.9s  [NbConvertApp] Converting notebook __script__.ipynb to html
   5341.6s  [NbConvertApp] Writing 284909 bytes to __results__.html
```

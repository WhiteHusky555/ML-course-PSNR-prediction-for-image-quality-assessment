# Лог прогона 06. Градиентный бустинг

Пояснения к прогону — в [README.md](README.md). Время в начале строки отсчитывается от запуска ядра.

```text
     14.4s  {"python": "3.13.18", "numpy": "2.1.3", "pandas": "2.3.3", "scikit-learn": "1.6.1", "scipy": "1.16.3", "scikit-image": "0.25.2", "opencv-python": "4.14.0.94", "opencv-python-headless": "4.14.0.94", "pyyaml": "6.0.3", "pytest": "8.4.2", "joblib": "1.6.0", "lightgbm": "4.6.0", "xgboost": "3.4.1", "catboost": "1.2.10"}
     45.9s  ................................................                         [100%]
     45.9s  =============================== warnings summary ===============================
     45.9s  ../../../usr/local/lib/python3.13/dist-packages/matplotlib/_fontconfig_pattern.py:64
     45.9s    /usr/local/lib/python3.13/dist-packages/matplotlib/_fontconfig_pattern.py:64: PyparsingDeprecationWarning: 'oneOf' deprecated - use 'one_of'
     45.9s      prop = Group((name + Suppress("=") + comma_separated(value)) | oneOf(_CONSTANTS))
     45.9s  
     45.9s  ../../../usr/local/lib/python3.13/dist-packages/matplotlib/_fontconfig_pattern.py:85
     45.9s  ../../../usr/local/lib/python3.13/dist-packages/matplotlib/_fontconfig_pattern.py:85
     45.9s  ../../../usr/local/lib/python3.13/dist-packages/matplotlib/_fontconfig_pattern.py:85
     45.9s  ../../../usr/local/lib/python3.13/dist-packages/matplotlib/_fontconfig_pattern.py:85
     45.9s  ../../../usr/local/lib/python3.13/dist-packages/matplotlib/_fontconfig_pattern.py:85
     45.9s  ../../../usr/local/lib/python3.13/dist-packages/matplotlib/_fontconfig_pattern.py:85
     45.9s    /usr/local/lib/python3.13/dist-packages/matplotlib/_fontconfig_pattern.py:85: PyparsingDeprecationWarning: 'parseString' deprecated - use 'parse_string'
     45.9s      parse = parser.parseString(pattern)
     45.9s  
     45.9s  ../../../usr/local/lib/python3.13/dist-packages/matplotlib/_fontconfig_pattern.py:89
     45.9s  ../../../usr/local/lib/python3.13/dist-packages/matplotlib/_fontconfig_pattern.py:89
     45.9s  ../../../usr/local/lib/python3.13/dist-packages/matplotlib/_fontconfig_pattern.py:89
     45.9s  ../../../usr/local/lib/python3.13/dist-packages/matplotlib/_fontconfig_pattern.py:89
     45.9s  ../../../usr/local/lib/python3.13/dist-packages/matplotlib/_fontconfig_pattern.py:89
     45.9s  ../../../usr/local/lib/python3.13/dist-packages/matplotlib/_fontconfig_pattern.py:89
     45.9s    /usr/local/lib/python3.13/dist-packages/matplotlib/_fontconfig_pattern.py:89: PyparsingDeprecationWarning: 'resetCache' deprecated - use 'reset_cache'
     45.9s      parser.resetCache()
     45.9s  
     45.9s  tests/test_models.py::test_boosting_models_fit_step_function[spec0]
     45.9s    /usr/local/lib/python3.13/dist-packages/sklearn/utils/validation.py:2739: UserWarning: X does not have valid feature names, but LGBMRegressor was fitted with feature names
     45.9s      warnings.warn(
     45.9s  
     45.9s  -- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
     45.9s  48 passed, 14 warnings in 25.75s
     47.9s  == check_config.yaml
    155.1s  catboost depth=4 loss_function=RMSE: MAE 0.7066
    266.6s  catboost depth=4 loss_function=MAE: MAE 0.6611
    370.3s  catboost depth=4 loss_function=Huber:delta=1.0: MAE 0.6712
    597.7s  catboost depth=6 loss_function=RMSE: MAE 0.6920
    833.0s  catboost depth=6 loss_function=MAE: MAE 0.6596
   1066.8s  catboost depth=6 loss_function=Huber:delta=1.0: MAE 0.6674
   1745.4s  catboost depth=8 loss_function=RMSE: MAE 0.6870
   2413.7s  catboost depth=8 loss_function=MAE: MAE 0.6698
   3100.2s  catboost depth=8 loss_function=Huber:delta=1.0: MAE 0.6689
   3129.3s  elasticnet alpha=0.0001 l1_ratio=0.1: MAE 0.8032
   3147.1s  elasticnet alpha=0.0001 l1_ratio=0.5: MAE 0.8025
   3161.7s  elasticnet alpha=0.0001 l1_ratio=0.9: MAE 0.8012
   3168.7s  elasticnet alpha=0.001 l1_ratio=0.1: MAE 0.8049
   3171.6s  elasticnet alpha=0.001 l1_ratio=0.5: MAE 0.8044
   3173.2s  elasticnet alpha=0.001 l1_ratio=0.9: MAE 0.8031
   3175.2s  elasticnet alpha=0.01 l1_ratio=0.1: MAE 0.8126
   3176.1s  elasticnet alpha=0.01 l1_ratio=0.5: MAE 0.8189
   3177.0s  elasticnet alpha=0.01 l1_ratio=0.9: MAE 0.8276
   3177.9s  elasticnet alpha=0.1 l1_ratio=0.1: MAE 0.8446
   3178.6s  elasticnet alpha=0.1 l1_ratio=0.5: MAE 0.8735
   3179.5s  elasticnet alpha=0.1 l1_ratio=0.9: MAE 0.9007
   3311.4s  forest max_features=sqrt min_samples_leaf=1: MAE 0.7028
   3419.4s  forest max_features=sqrt min_samples_leaf=3: MAE 0.6971
   3496.4s  forest max_features=sqrt min_samples_leaf=10: MAE 0.7029
   4007.7s  forest max_features=0.33 min_samples_leaf=1: MAE 0.6968
   4416.7s  forest max_features=0.33 min_samples_leaf=3: MAE 0.6883
   4737.2s  forest max_features=0.33 min_samples_leaf=10: MAE 0.6906
   5501.6s  forest max_features=0.5 min_samples_leaf=1: MAE 0.6970
   6125.0s  forest max_features=0.5 min_samples_leaf=3: MAE 0.6889
   6599.4s  forest max_features=0.5 min_samples_leaf=10: MAE 0.6902
   7643.5s  gradboost loss=squared_error max_depth=3: MAE 0.7272
   9334.1s  gradboost loss=squared_error max_depth=5: MAE 0.6982
  10371.2s  gradboost loss=absolute_error max_depth=3: MAE 0.6770
  12085.6s  gradboost loss=absolute_error max_depth=5: MAE 0.6628
  13099.1s  gradboost loss=huber max_depth=3: MAE 0.6866
  14763.6s  gradboost loss=huber max_depth=5: MAE 0.6749
  14829.4s  histgb loss=squared_error max_leaf_nodes=15: MAE 0.6943
  14939.9s  histgb loss=squared_error max_leaf_nodes=31: MAE 0.6853
  15148.0s  histgb loss=squared_error max_leaf_nodes=63: MAE 0.6850
  15228.0s  histgb loss=absolute_error max_leaf_nodes=15: MAE 0.6595
  15361.1s  histgb loss=absolute_error max_leaf_nodes=31: MAE 0.6566
  15609.3s  histgb loss=absolute_error max_leaf_nodes=63: MAE 0.6504
  15614.4s  knn metric=euclidean n_neighbors=3 weights=uniform: MAE 0.8101
  15619.5s  knn metric=euclidean n_neighbors=3 weights=distance: MAE 0.8091
  15624.6s  knn metric=euclidean n_neighbors=5 weights=uniform: MAE 0.7804
  15629.7s  knn metric=euclidean n_neighbors=5 weights=distance: MAE 0.7796
  15634.9s  knn metric=euclidean n_neighbors=10 weights=uniform: MAE 0.7672
  15640.0s  knn metric=euclidean n_neighbors=10 weights=distance: MAE 0.7659
  15645.2s  knn metric=euclidean n_neighbors=20 weights=uniform: MAE 0.7657
  15650.5s  knn metric=euclidean n_neighbors=20 weights=distance: MAE 0.7640
  15656.2s  knn metric=euclidean n_neighbors=50 weights=uniform: MAE 0.7778
  15661.9s  knn metric=euclidean n_neighbors=50 weights=distance: MAE 0.7753
  15668.3s  knn metric=euclidean n_neighbors=100 weights=uniform: MAE 0.7939
  15674.8s  knn metric=euclidean n_neighbors=100 weights=distance: MAE 0.7909
  15743.2s  knn metric=manhattan n_neighbors=3 weights=uniform: MAE 0.7927
  15811.6s  knn metric=manhattan n_neighbors=3 weights=distance: MAE 0.7916
  15880.4s  knn metric=manhattan n_neighbors=5 weights=uniform: MAE 0.7718
  15949.0s  knn metric=manhattan n_neighbors=5 weights=distance: MAE 0.7702
  16017.9s  knn metric=manhattan n_neighbors=10 weights=uniform: MAE 0.7533
  16087.2s  knn metric=manhattan n_neighbors=10 weights=distance: MAE 0.7519
  16157.3s  knn metric=manhattan n_neighbors=20 weights=uniform: MAE 0.7491
  16227.1s  knn metric=manhattan n_neighbors=20 weights=distance: MAE 0.7474
  16297.8s  knn metric=manhattan n_neighbors=50 weights=uniform: MAE 0.7642
  16368.0s  knn metric=manhattan n_neighbors=50 weights=distance: MAE 0.7613
  16440.1s  knn metric=manhattan n_neighbors=100 weights=uniform: MAE 0.7798
  16512.2s  knn metric=manhattan n_neighbors=100 weights=distance: MAE 0.7764
  16524.9s  lasso alpha=0.0001: MAE 0.8009
  16530.3s  lasso alpha=0.0003: MAE 0.8000
  16531.8s  lasso alpha=0.001: MAE 0.8025
  16532.7s  lasso alpha=0.003: MAE 0.8111
  16533.4s  lasso alpha=0.01: MAE 0.8301
  16534.0s  lasso alpha=0.03: MAE 0.8559
  16534.7s  lasso alpha=0.1: MAE 0.9083
  16535.4s  lasso alpha=0.3: MAE 1.0162
  16573.7s  lightgbm num_leaves=15 objective=l2: MAE 0.6931
  16617.0s  lightgbm num_leaves=15 objective=l1: MAE 0.6553
  16657.8s  lightgbm num_leaves=15 objective=huber: MAE 0.6617
  16723.5s  lightgbm num_leaves=31 objective=l2: MAE 0.6879
  16793.9s  lightgbm num_leaves=31 objective=l1: MAE 0.6537
  16861.0s  lightgbm num_leaves=31 objective=huber: MAE 0.6601
  16975.9s  lightgbm num_leaves=63 objective=l2: MAE 0.6869
  17097.4s  lightgbm num_leaves=63 objective=l1: MAE 0.6490
  17212.3s  lightgbm num_leaves=63 objective=huber: MAE 0.6635
  17213.4s  linear: MAE 0.8070
  17213.6s  median: MAE 1.0260
  17214.4s  ridge alpha=0.01: MAE 0.8060
  17215.3s  ridge alpha=0.1: MAE 0.8045
  17216.0s  ridge alpha=1.0: MAE 0.8034
  17216.8s  ridge alpha=10.0: MAE 0.8050
  17217.6s  ridge alpha=100.0: MAE 0.8110
  17218.4s  ridge alpha=1000.0: MAE 0.8306
  17219.3s  ridge alpha=10000.0: MAE 0.8891
  17223.1s  tree max_depth=3 min_samples_leaf=1: MAE 0.8513
  17226.8s  tree max_depth=3 min_samples_leaf=5: MAE 0.8513
  17230.4s  tree max_depth=3 min_samples_leaf=20: MAE 0.8513
  17234.2s  tree max_depth=3 min_samples_leaf=100: MAE 0.8502
  17240.3s  tree max_depth=5 min_samples_leaf=1: MAE 0.8076
  17246.3s  tree max_depth=5 min_samples_leaf=5: MAE 0.8115
  17252.1s  tree max_depth=5 min_samples_leaf=20: MAE 0.7991
  17257.7s  tree max_depth=5 min_samples_leaf=100: MAE 0.8060
  17268.9s  tree max_depth=10 min_samples_leaf=1: MAE 0.8243
  17279.7s  tree max_depth=10 min_samples_leaf=5: MAE 0.8169
  17289.9s  tree max_depth=10 min_samples_leaf=20: MAE 0.7817
  17298.2s  tree max_depth=10 min_samples_leaf=100: MAE 0.7801
  17316.8s  tree max_depth=20 min_samples_leaf=1: MAE 0.9193
  17332.6s  tree max_depth=20 min_samples_leaf=5: MAE 0.8783
  17345.3s  tree max_depth=20 min_samples_leaf=20: MAE 0.7966
  17354.4s  tree max_depth=20 min_samples_leaf=100: MAE 0.7801
  17377.0s  tree max_depth=None min_samples_leaf=1: MAE 0.9362
  17394.8s  tree max_depth=None min_samples_leaf=5: MAE 0.8835
  17408.1s  tree max_depth=None min_samples_leaf=20: MAE 0.7968
  17417.3s  tree max_depth=None min_samples_leaf=100: MAE 0.7801
  17493.8s  xgboost max_depth=4 objective=reg:squarederror: MAE 0.6999
  17571.3s  xgboost max_depth=4 objective=reg:absoluteerror: MAE 0.6573
  17646.4s  xgboost max_depth=4 objective=reg:pseudohubererror: MAE 0.6667
  17828.0s  xgboost max_depth=6 objective=reg:squarederror: MAE 0.6826
  17997.9s  xgboost max_depth=6 objective=reg:absoluteerror: MAE 0.6462
  18167.6s  xgboost max_depth=6 objective=reg:pseudohubererror: MAE 0.6657
  18607.6s  xgboost max_depth=8 objective=reg:squarederror: MAE 0.6757
  18988.7s  xgboost max_depth=8 objective=reg:absoluteerror: MAE 0.6456
  19345.4s  xgboost max_depth=8 objective=reg:pseudohubererror: MAE 0.6730
  19423.3s                                                  model      mae  fold_std    train_mae    mae_1    mae_2    mae_3    mae_4    mae_5
  19423.3s        xgboost max_depth=8 objective=reg:absoluteerror 0.645552  0.008294 1.727994e-01 0.965504 0.090532 0.382378 0.700832 1.088514
  19423.3s        xgboost max_depth=6 objective=reg:absoluteerror 0.646247  0.008446 2.979362e-01 0.961895 0.094347 0.393698 0.701523 1.079775
  19423.3s                    lightgbm num_leaves=63 objective=l1 0.648960  0.011838 3.154735e-01 0.976508 0.085915 0.401652 0.683706 1.097021
  19423.3s           histgb loss=absolute_error max_leaf_nodes=63 0.650416  0.011960 3.084958e-01 0.977639 0.086729 0.404899 0.683219 1.099595
  19423.3s                    lightgbm num_leaves=31 objective=l1 0.653682  0.012926 4.015867e-01 0.979146 0.086387 0.426071 0.687992 1.088815
  19423.3s                    lightgbm num_leaves=15 objective=l1 0.655301  0.010830 5.034263e-01 0.978118 0.087139 0.432665 0.700624 1.077959
  19423.3s           histgb loss=absolute_error max_leaf_nodes=31 0.656595  0.012614 4.082874e-01 0.980523 0.087118 0.436113 0.690946 1.088275
  19423.3s        xgboost max_depth=4 objective=reg:absoluteerror 0.657262  0.009113 4.810056e-01 0.972090 0.097101 0.427188 0.712562 1.077370
  19423.3s           histgb loss=absolute_error max_leaf_nodes=15 0.659546  0.012573 5.055839e-01 0.986824 0.087594 0.439575 0.702482 1.081255
  19423.3s                     catboost depth=6 loss_function=MAE 0.659616  0.011548 4.205432e-01 0.993994 0.090508 0.418940 0.705013 1.089626
  19423.3s                 lightgbm num_leaves=31 objective=huber 0.660142  0.008358 2.688681e-01 0.976425 0.102876 0.399320 0.716552 1.105536
  19423.3s                     catboost depth=4 loss_function=MAE 0.661103  0.010851 5.343114e-01 0.996438 0.089574 0.431004 0.707647 1.080852
  19423.3s                 lightgbm num_leaves=15 objective=huber 0.661700  0.009467 4.325979e-01 0.973944 0.103409 0.425371 0.710473 1.095303
  19423.3s              gradboost loss=absolute_error max_depth=5 0.662836  0.010281 4.885741e-01 0.985140 0.088852 0.438275 0.707502 1.094414
  19423.3s                 lightgbm num_leaves=63 objective=huber 0.663505  0.005443 1.295490e-01 0.976658 0.102724 0.381922 0.750501 1.105721
  19423.3s     xgboost max_depth=6 objective=reg:pseudohubererror 0.665679  0.005668 1.514403e-01 0.967033 0.103443 0.374832 0.774042 1.109047
  19423.3s     xgboost max_depth=4 objective=reg:pseudohubererror 0.666727  0.007557 3.657928e-01 0.983617 0.105430 0.390235 0.747692 1.106662
  19423.3s         catboost depth=6 loss_function=Huber:delta=1.0 0.667422  0.007033 2.838621e-01 0.995704 0.109438 0.387265 0.732538 1.112163
  19423.3s         catboost depth=8 loss_function=Huber:delta=1.0 0.668901  0.010531 1.732272e-01 0.997573 0.109466 0.379539 0.739104 1.118825
  19423.3s                     catboost depth=8 loss_function=MAE 0.669767  0.010438 2.640107e-01 1.007282 0.093657 0.419008 0.719553 1.109335
  19423.3s         catboost depth=4 loss_function=Huber:delta=1.0 0.671174  0.010639 4.213085e-01 0.998124 0.111843 0.396812 0.737041 1.112048
  19423.3s     xgboost max_depth=8 objective=reg:pseudohubererror 0.673021  0.009416 4.863965e-02 0.982900 0.101187 0.368438 0.796738 1.115842
  19423.3s                       gradboost loss=huber max_depth=5 0.674856  0.005262 2.430680e-01 0.990538 0.105408 0.391575 0.768433 1.118326
  19423.3s         xgboost max_depth=8 objective=reg:squarederror 0.675678  0.004406 3.081023e-02 0.985497 0.099306 0.375508 0.806012 1.112066
  19423.3s              gradboost loss=absolute_error max_depth=3 0.676968  0.012902 5.874419e-01 1.014371 0.089201 0.451820 0.718928 1.110521
  19423.3s         xgboost max_depth=6 objective=reg:squarederror 0.682616  0.006065 1.397318e-01 0.984285 0.107036 0.381395 0.821863 1.118502
  19423.3s            histgb loss=squared_error max_leaf_nodes=63 0.685004  0.005945 6.829604e-02 0.991731 0.108136 0.380219 0.824153 1.120783
  19423.3s            histgb loss=squared_error max_leaf_nodes=31 0.685278  0.004014 1.833066e-01 0.993927 0.110278 0.382226 0.817059 1.122899
  19423.3s                       gradboost loss=huber max_depth=3 0.686622  0.008523 4.824578e-01 1.006506 0.113024 0.430355 0.764450 1.118772
  19423.3s                    lightgbm num_leaves=63 objective=l2 0.686886  0.007455 6.874940e-02 0.988710 0.108019 0.383005 0.837715 1.116983
  19423.3s                    catboost depth=8 loss_function=RMSE 0.686950  0.008859 1.354906e-01 1.011655 0.110495 0.375877 0.803507 1.133218
  19423.3s                    lightgbm num_leaves=31 objective=l2 0.687907  0.008560 1.814982e-01 0.990385 0.109708 0.388295 0.830256 1.120893
  19423.3s            forest max_features=0.33 min_samples_leaf=3 0.688301  0.004470 3.196000e-01 1.015395 0.096432 0.376158 0.837132 1.116390
  19423.3s             forest max_features=0.5 min_samples_leaf=3 0.688929  0.004476 3.145947e-01 1.014377 0.097010 0.375475 0.842975 1.114807
  19423.3s            forest max_features=0.5 min_samples_leaf=10 0.690194  0.005725 4.795158e-01 1.021757 0.096791 0.385304 0.831107 1.116013
  19423.3s           forest max_features=0.33 min_samples_leaf=10 0.690570  0.005707 4.880143e-01 1.022198 0.097056 0.383023 0.831004 1.119567
  19423.3s                    catboost depth=6 loss_function=RMSE 0.691955  0.007066 2.664718e-01 1.005022 0.116987 0.383274 0.821194 1.133298
  19423.3s                    lightgbm num_leaves=15 objective=l2 0.693076  0.008703 3.395383e-01 0.997646 0.111683 0.395725 0.837252 1.123073
  19423.3s            histgb loss=squared_error max_leaf_nodes=15 0.694276  0.006847 3.440614e-01 1.002562 0.112388 0.394975 0.832821 1.128636
  19423.3s            forest max_features=0.33 min_samples_leaf=1 0.696756  0.004156 2.533935e-01 1.018466 0.099712 0.377524 0.868860 1.119219
  19423.3s             forest max_features=0.5 min_samples_leaf=1 0.696958  0.005464 2.532194e-01 1.016157 0.099834 0.376156 0.878046 1.114595
  19423.3s            forest max_features=sqrt min_samples_leaf=3 0.697087  0.004653 3.507340e-01 1.030016 0.099329 0.387728 0.843712 1.124650
  19423.3s               gradboost loss=squared_error max_depth=5 0.698212  0.004477 2.154311e-01 0.996307 0.118315 0.392597 0.855303 1.128540
  19423.3s         xgboost max_depth=4 objective=reg:squarederror 0.699875  0.008106 3.836997e-01 1.004261 0.117616 0.394981 0.852137 1.130381
  19423.3s            forest max_features=sqrt min_samples_leaf=1 0.702779  0.003807 2.569424e-01 1.034239 0.100578 0.384052 0.868462 1.126563
  19423.3s           forest max_features=sqrt min_samples_leaf=10 0.702869  0.004481 5.322876e-01 1.043774 0.100247 0.392866 0.846244 1.131217
  19423.3s                    catboost depth=4 loss_function=RMSE 0.706607  0.006391 4.310097e-01 1.016332 0.124416 0.384959 0.862172 1.145157
  19423.3s               gradboost loss=squared_error max_depth=3 0.727246  0.005021 4.902768e-01 1.028892 0.129902 0.418929 0.910415 1.148089
  19423.3s   knn metric=manhattan n_neighbors=20 weights=distance 0.747363  0.008019 0.000000e+00 1.133950 0.116569 0.421032 0.849401 1.215865
  19423.3s    knn metric=manhattan n_neighbors=20 weights=uniform 0.749064  0.007656 7.032280e-01 1.135713 0.116666 0.422139 0.852599 1.218201
  19423.3s   knn metric=manhattan n_neighbors=10 weights=distance 0.751947  0.009924 0.000000e+00 1.135753 0.117740 0.422783 0.856412 1.227049
  19423.3s    knn metric=manhattan n_neighbors=10 weights=uniform 0.753322  0.009923 6.622242e-01 1.136052 0.117673 0.423784 0.859412 1.229690
  19423.3s   knn metric=manhattan n_neighbors=50 weights=distance 0.761334  0.006818 0.000000e+00 1.144551 0.116551 0.432175 0.875471 1.237921
  19423.3s   knn metric=euclidean n_neighbors=20 weights=distance 0.763962  0.006223 3.038490e-07 1.157616 0.118101 0.433287 0.863701 1.247103
  19423.3s    knn metric=manhattan n_neighbors=50 weights=uniform 0.764175  0.006656 7.435328e-01 1.147257 0.116827 0.434477 0.880969 1.241348
  19423.3s    knn metric=euclidean n_neighbors=20 weights=uniform 0.765672  0.005971 7.195662e-01 1.158776 0.118158 0.434144 0.866981 1.250301
  19423.3s   knn metric=euclidean n_neighbors=10 weights=distance 0.765947  0.007036 1.540228e-07 1.166860 0.120813 0.436762 0.864710 1.240589
  19423.3s    knn metric=euclidean n_neighbors=10 weights=uniform 0.767151  0.006571 6.793696e-01 1.167331 0.120791 0.437634 0.867254 1.242745
  19423.3s    knn metric=manhattan n_neighbors=5 weights=distance 0.770241  0.013204 0.000000e+00 1.166741 0.121725 0.434145 0.867752 1.260842
  19423.3s     knn metric=manhattan n_neighbors=5 weights=uniform 0.771818  0.013397 5.943856e-01 1.167782 0.121581 0.435563 0.870841 1.263322
  19423.3s   knn metric=euclidean n_neighbors=50 weights=distance 0.775259  0.007546 7.255150e-07 1.168991 0.117824 0.438175 0.883565 1.267741
  19423.3s  knn metric=manhattan n_neighbors=100 weights=distance 0.776385  0.007233 0.000000e+00 1.164867 0.116830 0.441239 0.901838 1.257153
  19423.3s    knn metric=euclidean n_neighbors=50 weights=uniform 0.777781  0.007654 7.578610e-01 1.171621 0.118061 0.439820 0.888067 1.271333
  19423.3s    knn metric=euclidean n_neighbors=5 weights=distance 0.779602  0.010474 7.450185e-08 1.188165 0.123459 0.446941 0.883410 1.256036
  19423.3s   knn metric=manhattan n_neighbors=100 weights=uniform 0.779791  0.007188 7.684952e-01 1.168610 0.117301 0.443879 0.908042 1.261122
  19423.3s               tree max_depth=None min_samples_leaf=100 0.780080  0.012073 7.253160e-01 1.124658 0.126496 0.465024 0.961867 1.222356
  19423.3s                 tree max_depth=20 min_samples_leaf=100 0.780080  0.012073 7.253160e-01 1.124658 0.126496 0.465024 0.961867 1.222356
  19423.3s                 tree max_depth=10 min_samples_leaf=100 0.780106  0.012413 7.261002e-01 1.124658 0.125714 0.465933 0.961867 1.222356
  19423.3s     knn metric=euclidean n_neighbors=5 weights=uniform 0.780353  0.010243 6.128966e-01 1.187520 0.123103 0.448293 0.885624 1.257224
  19423.3s                  tree max_depth=10 min_samples_leaf=20 0.781733  0.009909 6.239731e-01 1.130677 0.115926 0.462600 0.948860 1.250601
  19423.3s  knn metric=euclidean n_neighbors=100 weights=distance 0.790908  0.008088 1.386155e-06 1.185239 0.117824 0.446790 0.914409 1.290278
  19423.3s    knn metric=manhattan n_neighbors=3 weights=distance 0.791607  0.013257 0.000000e+00 1.197811 0.125614 0.439543 0.895296 1.299769
  19423.3s     knn metric=manhattan n_neighbors=3 weights=uniform 0.792671  0.013441 5.103679e-01 1.199093 0.125341 0.440159 0.897770 1.300990
  19423.3s   knn metric=euclidean n_neighbors=100 weights=uniform 0.793887  0.008179 7.826545e-01 1.188344 0.118103 0.449049 0.920007 1.293932
  19423.3s                  tree max_depth=20 min_samples_leaf=20 0.796604  0.015152 5.782544e-01 1.151554 0.116219 0.460065 0.958644 1.296537
  19423.3s                tree max_depth=None min_samples_leaf=20 0.796793  0.015128 5.778456e-01 1.151554 0.116803 0.460577 0.958493 1.296537
  19423.3s                   tree max_depth=5 min_samples_leaf=20 0.799139  0.009415 7.586327e-01 1.168008 0.137094 0.468763 0.959689 1.262143
  19423.3s                                     lasso alpha=0.0003 0.800028  0.013042 7.867549e-01 1.091709 0.213111 0.475232 1.048308 1.171780
  19423.3s                                     lasso alpha=0.0001 0.800922  0.013506 7.853148e-01 1.083071 0.214902 0.480945 1.059119 1.166572
  19423.3s                   elasticnet alpha=0.0001 l1_ratio=0.9 0.801201  0.013581 7.858737e-01 1.085023 0.216338 0.479597 1.058176 1.166872
  19423.3s                                      lasso alpha=0.001 0.802498  0.014340 7.917772e-01 1.106151 0.213724 0.477635 1.035350 1.179630
  19423.3s                   elasticnet alpha=0.0001 l1_ratio=0.5 0.802512  0.013544 7.875302e-01 1.088171 0.219032 0.479161 1.058944 1.167252
  19423.3s                    elasticnet alpha=0.001 l1_ratio=0.9 0.803131  0.014066 7.922401e-01 1.106442 0.214749 0.477891 1.036782 1.179790
  19423.3s                   elasticnet alpha=0.0001 l1_ratio=0.1 0.803225  0.013685 7.883906e-01 1.089471 0.220532 0.478826 1.059816 1.167482
  19423.3s                                        ridge alpha=1.0 0.803362  0.013718 7.885078e-01 1.089447 0.220828 0.478973 1.060132 1.167429
  19423.3s                    elasticnet alpha=0.001 l1_ratio=0.5 0.804450  0.013574 7.928440e-01 1.106615 0.217594 0.478085 1.040210 1.179743
  19423.3s                                        ridge alpha=0.1 0.804484  0.013732 7.876758e-01 1.081582 0.220077 0.487088 1.067914 1.165761
  19423.3s                    elasticnet alpha=0.001 l1_ratio=0.1 0.804929  0.013621 7.929727e-01 1.105107 0.220273 0.477833 1.043302 1.178128
  19423.3s                                       ridge alpha=10.0 0.805031  0.013639 7.929351e-01 1.104317 0.221044 0.477574 1.044873 1.177346
  19423.3s                  tree max_depth=5 min_samples_leaf=100 0.805986  0.011048 7.726339e-01 1.159791 0.152960 0.482268 0.966927 1.267982
  19423.3s                                       ridge alpha=0.01 0.806034  0.013894 7.885506e-01 1.079347 0.220151 0.491783 1.073080 1.165808
  19423.3s                                                 linear 0.807000  0.013944 7.893295e-01 1.079105 0.221555 0.493807 1.074868 1.165668
  19423.3s                    tree max_depth=5 min_samples_leaf=1 0.807618  0.006543 7.559056e-01 1.183974 0.138466 0.474351 0.969324 1.271973
  19423.3s    knn metric=euclidean n_neighbors=3 weights=distance 0.809087  0.012228 4.139641e-08 1.245625 0.129965 0.448885 0.918116 1.302842
  19423.3s     knn metric=euclidean n_neighbors=3 weights=uniform 0.810075  0.011922 5.337346e-01 1.247213 0.130131 0.448991 0.920519 1.303521
  19423.3s                                      ridge alpha=100.0 0.810965  0.012437 8.017750e-01 1.125982 0.219200 0.490436 1.017653 1.201556
  19423.3s                                      lasso alpha=0.003 0.811139  0.015039 8.030606e-01 1.121533 0.223147 0.495465 1.019703 1.195849
  19423.3s                    tree max_depth=5 min_samples_leaf=5 0.811519  0.010801 7.568641e-01 1.177482 0.138560 0.475206 0.984593 1.281756
  19423.3s                     elasticnet alpha=0.01 l1_ratio=0.1 0.812616  0.012923 8.041716e-01 1.129072 0.222323 0.494089 1.012317 1.205279
  19423.3s                   tree max_depth=10 min_samples_leaf=5 0.816851  0.008610 5.383784e-01 1.177763 0.120665 0.477633 0.995951 1.312244
  19423.3s                     elasticnet alpha=0.01 l1_ratio=0.5 0.818941  0.015013 8.123585e-01 1.139073 0.227113 0.509684 1.001130 1.217704
  19423.3s                   tree max_depth=10 min_samples_leaf=1 0.824346  0.008239 5.021226e-01 1.198455 0.123215 0.490394 0.993779 1.315888
  19423.3s                     elasticnet alpha=0.01 l1_ratio=0.9 0.827610  0.015016 8.220424e-01 1.151325 0.231208 0.523157 1.001502 1.230857
  19423.3s                                       lasso alpha=0.01 0.830111  0.014597 8.247036e-01 1.155011 0.232614 0.526358 1.002286 1.234286
  19423.3s                                     ridge alpha=1000.0 0.830560  0.012188 8.248700e-01 1.163510 0.206915 0.512995 1.017197 1.252184
  19423.3s                      elasticnet alpha=0.1 l1_ratio=0.1 0.844604  0.012009 8.404939e-01 1.188861 0.209128 0.533528 1.014647 1.276858
  19423.3s                  tree max_depth=3 min_samples_leaf=100 0.850185  0.004316 8.382491e-01 1.228124 0.194364 0.518911 0.957889 1.351635
  19423.3s                    tree max_depth=3 min_samples_leaf=5 0.851275  0.004845 8.385166e-01 1.228120 0.195525 0.518923 0.962170 1.351635
  19423.3s                    tree max_depth=3 min_samples_leaf=1 0.851275  0.004845 8.385166e-01 1.228120 0.195525 0.518923 0.962170 1.351635
  19423.3s                   tree max_depth=3 min_samples_leaf=20 0.851275  0.004845 8.385166e-01 1.228120 0.195525 0.518923 0.962170 1.351635
  19423.3s                                       lasso alpha=0.03 0.855877  0.012807 8.524598e-01 1.193797 0.226232 0.553905 1.029512 1.275940
  19423.3s                      elasticnet alpha=0.1 l1_ratio=0.5 0.873549  0.010367 8.711784e-01 1.231434 0.225407 0.546756 1.044519 1.319630
  19423.3s                   tree max_depth=20 min_samples_leaf=5 0.878346  0.011666 3.204389e-01 1.280478 0.125253 0.487694 1.076543 1.421761
  19423.3s                 tree max_depth=None min_samples_leaf=5 0.883464  0.010677 3.113946e-01 1.285861 0.131586 0.495354 1.074244 1.430277
  19423.3s                                    ridge alpha=10000.0 0.889124  0.009074 8.866370e-01 1.284852 0.175402 0.517071 1.103596 1.364698
  19423.3s                      elasticnet alpha=0.1 l1_ratio=0.9 0.900679  0.008026 8.985084e-01 1.264809 0.252499 0.531030 1.090104 1.364953
  19423.3s                                        lasso alpha=0.1 0.908265  0.007730 9.062888e-01 1.275769 0.258662 0.528217 1.102683 1.375993
  19423.3s                   tree max_depth=20 min_samples_leaf=1 0.919282  0.019100 6.273259e-02 1.333519 0.138318 0.479512 1.155809 1.489255
  19423.3s                 tree max_depth=None min_samples_leaf=1 0.936218  0.020356 5.733800e-12 1.347419 0.150124 0.502028 1.150073 1.531447
  19423.3s                                        lasso alpha=0.3 1.016237  0.007882 1.015961e+00 1.326304 0.473927 0.567346 1.249612 1.463999
  19423.3s                                                 median 1.025969  0.018641 1.025906e+00 1.662687 0.185335 0.576075 0.769100 1.936646
  19424.1s  == check_categorical.yaml
  19429.6s  no_variant alpha=0.1: MAE 0.8071
  19430.3s  no_variant alpha=1.0: MAE 0.8069
  19431.0s  no_variant alpha=10.0: MAE 0.8092
  19431.6s  no_variant alpha=100.0: MAE 0.8154
  19432.7s  onehot alpha=0.1: MAE 0.7863
  19433.6s  onehot alpha=1.0: MAE 0.7869
  19434.4s  onehot alpha=10.0: MAE 0.7834
  19435.5s  onehot alpha=100.0: MAE 0.7860
  19436.3s  ordinal alpha=0.1: MAE 0.8065
  19437.0s  ordinal alpha=1.0: MAE 0.8062
  19437.7s  ordinal alpha=10.0: MAE 0.8081
  19438.3s  ordinal alpha=100.0: MAE 0.8146
  19439.1s  target alpha=0.1: MAE 0.7930
  19439.9s  target alpha=1.0: MAE 0.7920
  19440.7s  target alpha=10.0: MAE 0.7902
  19441.6s  target alpha=100.0: MAE 0.7979
  19441.7s                   model      mae  fold_std  train_mae    mae_1    mae_2    mae_3    mae_4    mae_5
  19441.7s       onehot alpha=10.0 0.783389  0.010372   0.767646 1.088666 0.225829 0.462769 0.991971 1.147713
  19441.7s      onehot alpha=100.0 0.786035  0.012700   0.774417 1.115766 0.205932 0.474990 0.959062 1.174423
  19441.7s        onehot alpha=0.1 0.786312  0.010924   0.766329 1.071359 0.244518 0.471747 1.005893 1.138045
  19441.7s        onehot alpha=1.0 0.786868  0.010529   0.768165 1.078160 0.242500 0.467480 1.006455 1.139745
  19441.7s       target alpha=10.0 0.790218  0.013893   0.774645 1.094595 0.221498 0.465650 1.022613 1.146736
  19441.7s        target alpha=1.0 0.792013  0.014469   0.773574 1.084065 0.226519 0.469213 1.043958 1.136311
  19441.7s        target alpha=0.1 0.793023  0.014540   0.773095 1.077326 0.227355 0.477445 1.047838 1.135153
  19441.7s      target alpha=100.0 0.797907  0.013425   0.785838 1.118658 0.213251 0.487904 0.990854 1.178869
  19441.7s       ordinal alpha=1.0 0.806151  0.017212   0.787763 1.088654 0.227372 0.484181 1.070470 1.160078
  19441.7s       ordinal alpha=0.1 0.806453  0.016671   0.786425 1.079719 0.227669 0.492870 1.074547 1.157458
  19441.7s    no_variant alpha=1.0 0.806939  0.017193   0.788594 1.088716 0.226025 0.483459 1.071249 1.165243
  19441.7s    no_variant alpha=0.1 0.807091  0.016762   0.787275 1.079694 0.226247 0.492170 1.074822 1.162520
  19441.7s      ordinal alpha=10.0 0.808063  0.017541   0.792455 1.103438 0.225641 0.482096 1.057899 1.171240
  19441.7s   no_variant alpha=10.0 0.809195  0.017448   0.793540 1.103341 0.224811 0.481295 1.059340 1.177189
  19441.7s     ordinal alpha=100.0 0.814649  0.015659   0.802102 1.125836 0.220786 0.495794 1.032199 1.198631
  19441.7s  no_variant alpha=100.0 0.815379  0.015562   0.802856 1.125682 0.220627 0.495411 1.032992 1.202185
  19442.4s  == check_preprocessing.yaml
  19447.9s  clip_01 alpha=0.1: MAE 0.8045
  19448.8s  clip_01 alpha=1.0: MAE 0.8034
  19449.5s  clip_01 alpha=10.0: MAE 0.8050
  19450.2s  clip_01 alpha=100.0: MAE 0.8110
  19451.0s  clip_1 alpha=0.1: MAE 0.8096
  19451.9s  clip_1 alpha=1.0: MAE 0.8075
  19452.6s  clip_1 alpha=10.0: MAE 0.8083
  19453.5s  clip_1 alpha=100.0: MAE 0.8130
  19455.5s  linear_select_95: MAE 0.8511
  19456.5s  linear_standard: MAE 0.8100
  19456.6s  median: MAE 1.0260
  19457.2s  pca_95 alpha=0.1: MAE 0.9285
  19457.6s  pca_95 alpha=1.0: MAE 0.9285
  19458.1s  pca_95 alpha=10.0: MAE 0.9284
  19458.6s  pca_95 alpha=100.0: MAE 0.9282
  19459.2s  pca_99 alpha=0.1: MAE 0.8545
  19459.7s  pca_99 alpha=1.0: MAE 0.8544
  19460.3s  pca_99 alpha=10.0: MAE 0.8543
  19460.8s  pca_99 alpha=100.0: MAE 0.8528
  19467.4s  quantile alpha=0.1: MAE 0.8305
  19473.8s  quantile alpha=1.0: MAE 0.8299
  19480.3s  quantile alpha=10.0: MAE 0.8305
  19486.8s  quantile alpha=100.0: MAE 0.8344
  19487.2s  raw alpha=0.1: MAE 0.8110
  19487.7s  raw alpha=1.0: MAE 0.8209
  19488.3s  raw alpha=10.0: MAE 0.8372
  19488.8s  raw alpha=100.0: MAE 0.8654
  19489.6s  robust alpha=0.1: MAE 0.8080
  19490.5s  robust alpha=1.0: MAE 0.8090
  19491.4s  robust alpha=10.0: MAE 0.8131
  19492.4s  robust alpha=100.0: MAE 0.8243
  19493.8s  select_90 alpha=0.1: MAE 0.8673
  19495.3s  select_90 alpha=1.0: MAE 0.8672
  19496.8s  select_90 alpha=10.0: MAE 0.8665
  19498.2s  select_90 alpha=100.0: MAE 0.8623
  19499.7s  select_95 alpha=0.1: MAE 0.8510
  19501.2s  select_95 alpha=1.0: MAE 0.8506
  19502.8s  select_95 alpha=10.0: MAE 0.8476
  19504.3s  select_95 alpha=100.0: MAE 0.8452
  19505.9s  select_99 alpha=0.1: MAE 0.8131
  19507.5s  select_99 alpha=1.0: MAE 0.8117
  19509.0s  select_99 alpha=10.0: MAE 0.8131
  19510.6s  select_99 alpha=100.0: MAE 0.8173
  19511.3s  standard alpha=0.1: MAE 0.8071
  19512.0s  standard alpha=1.0: MAE 0.8069
  19512.6s  standard alpha=10.0: MAE 0.8092
  19513.2s  standard alpha=100.0: MAE 0.8154
  19513.4s                  model      mae  fold_std  train_mae    mae_1    mae_2    mae_3    mae_4    mae_5
  19513.4s      clip_01 alpha=1.0 0.803362  0.013718   0.788508 1.089447 0.220828 0.478973 1.060132 1.167429
  19513.4s      clip_01 alpha=0.1 0.804484  0.013732   0.787676 1.081582 0.220077 0.487088 1.067914 1.165761
  19513.4s     clip_01 alpha=10.0 0.805031  0.013639   0.792935 1.104317 0.221044 0.477574 1.044873 1.177346
  19513.4s     standard alpha=1.0 0.806939  0.017193   0.788594 1.088716 0.226025 0.483459 1.071249 1.165243
  19513.4s     standard alpha=0.1 0.807091  0.016762   0.787275 1.079694 0.226247 0.492170 1.074822 1.162520
  19513.4s       clip_1 alpha=1.0 0.807515  0.012439   0.793017 1.093998 0.218040 0.480313 1.069235 1.175990
  19513.4s       robust alpha=0.1 0.807961  0.016810   0.788408 1.083616 0.229261 0.489068 1.074693 1.163165
  19513.4s      clip_1 alpha=10.0 0.808268  0.011426   0.796198 1.107190 0.219751 0.478217 1.052734 1.183448
  19513.4s       robust alpha=1.0 0.808981  0.017328   0.791471 1.093936 0.230410 0.481863 1.069386 1.169308
  19513.4s    standard alpha=10.0 0.809195  0.017448   0.793540 1.103341 0.224811 0.481295 1.059340 1.177189
  19513.4s       clip_1 alpha=0.1 0.809579  0.012994   0.793469 1.086582 0.219212 0.489203 1.078333 1.174563
  19513.4s        linear_standard 0.810013  0.017793   0.790053 1.076826 0.229050 0.501137 1.080486 1.162565
  19513.4s          raw alpha=0.1 0.810963  0.019359   0.792965 1.091133 0.233746 0.491258 1.068390 1.170288
  19513.4s    clip_01 alpha=100.0 0.810965  0.012437   0.801775 1.125982 0.219200 0.490436 1.017653 1.201556
  19513.4s    select_99 alpha=1.0 0.811732  0.015227   0.795128 1.087885 0.228829 0.490112 1.078390 1.173443
  19513.4s     clip_1 alpha=100.0 0.813042  0.010581   0.804443 1.128534 0.219160 0.490797 1.021409 1.205311
  19513.4s   select_99 alpha=10.0 0.813101  0.016700   0.798442 1.100433 0.229118 0.490548 1.067249 1.178156
  19513.4s    select_99 alpha=0.1 0.813132  0.013748   0.795793 1.085560 0.229867 0.493469 1.081684 1.175081
  19513.4s      robust alpha=10.0 0.813133  0.016357   0.798048 1.111764 0.227353 0.490346 1.049396 1.186808
  19513.4s   standard alpha=100.0 0.815379  0.015562   0.802856 1.125682 0.220627 0.495411 1.032992 1.202185
  19513.4s  select_99 alpha=100.0 0.817269  0.015649   0.805453 1.123889 0.225737 0.506196 1.031276 1.199247
  19513.4s          raw alpha=1.0 0.820944  0.019732   0.803995 1.108251 0.237460 0.508367 1.064558 1.186084
  19513.4s     robust alpha=100.0 0.824326  0.014970   0.812397 1.134939 0.224831 0.512584 1.034015 1.215262
  19513.4s     quantile alpha=1.0 0.829924  0.006831   0.812649 1.062733 0.264255 0.581870 1.032019 1.208741
  19513.4s    quantile alpha=10.0 0.830451  0.006341   0.814858 1.064359 0.270650 0.580293 1.024593 1.212358
  19513.4s     quantile alpha=0.1 0.830521  0.007074   0.812900 1.062928 0.263213 0.582779 1.036035 1.207651
  19513.4s   quantile alpha=100.0 0.834426  0.006728   0.822651 1.079549 0.276350 0.574842 1.016145 1.225243
  19513.4s         raw alpha=10.0 0.837214  0.018304   0.821813 1.136766 0.242884 0.525532 1.065396 1.215491
  19513.4s  select_95 alpha=100.0 0.845234  0.014261   0.835308 1.141273 0.256711 0.546809 1.039815 1.241562
  19513.4s   select_95 alpha=10.0 0.847602  0.016235   0.835725 1.133386 0.260735 0.525823 1.077604 1.240464
  19513.4s    select_95 alpha=1.0 0.850597  0.016965   0.838100 1.132631 0.261511 0.526129 1.091461 1.241252
  19513.4s    select_95 alpha=0.1 0.851038  0.017079   0.838465 1.132578 0.261601 0.526396 1.093248 1.241369
  19513.4s       linear_select_95 0.851090  0.017092   0.838508 1.132573 0.261611 0.526429 1.093455 1.241382
  19513.4s     pca_99 alpha=100.0 0.852848  0.012068   0.847016 1.150645 0.246447 0.561076 1.058263 1.247811
  19513.4s      pca_99 alpha=10.0 0.854279  0.012254   0.848244 1.149670 0.250513 0.564456 1.060983 1.245772
  19513.4s       pca_99 alpha=1.0 0.854441  0.012275   0.848388 1.149596 0.250944 0.564825 1.061271 1.245571
  19513.4s       pca_99 alpha=0.1 0.854458  0.012277   0.848402 1.149589 0.250987 0.564863 1.061300 1.245551
  19513.4s  select_90 alpha=100.0 0.862269  0.009189   0.853412 1.184441 0.229357 0.538702 1.088948 1.269896
  19513.4s        raw alpha=100.0 0.865362  0.014982   0.852922 1.182207 0.260310 0.550669 1.089804 1.243822
  19513.4s   select_90 alpha=10.0 0.866450  0.009550   0.856165 1.174969 0.242323 0.540983 1.108687 1.265288
  19513.4s    select_90 alpha=1.0 0.867246  0.009657   0.856751 1.173892 0.244550 0.541493 1.111620 1.264674
  19513.4s    select_90 alpha=0.1 0.867333  0.009669   0.856816 1.173785 0.244793 0.541551 1.111930 1.264606
  19513.4s     pca_95 alpha=100.0 0.928225  0.009874   0.923501 1.372394 0.235610 0.551732 1.106996 1.374396
  19513.4s      pca_95 alpha=10.0 0.928438  0.009910   0.923662 1.372710 0.236581 0.552236 1.106674 1.373988
  19513.4s       pca_95 alpha=1.0 0.928460  0.009914   0.923679 1.372743 0.236679 0.552288 1.106642 1.373947
  19513.4s       pca_95 alpha=0.1 0.928462  0.009915   0.923681 1.372746 0.236689 0.552293 1.106639 1.373942
  19513.4s                 median 1.025969  0.018641   1.025906 1.662687 0.185335 0.576075 0.769100 1.936646
  19514.1s  == check_validation.yaml
  20016.4s  forest min_samples_leaf=1: MAE 0.6913
  20417.8s  forest min_samples_leaf=3: MAE 0.6842
  20721.2s  forest min_samples_leaf=10: MAE 0.6865
  20726.3s  knn metric=euclidean n_neighbors=3 weights=uniform: MAE 0.7798
  20731.4s  knn metric=euclidean n_neighbors=3 weights=distance: MAE 0.7756
  20736.5s  knn metric=euclidean n_neighbors=5 weights=uniform: MAE 0.7692
  20741.7s  knn metric=euclidean n_neighbors=5 weights=distance: MAE 0.7639
  20746.8s  knn metric=euclidean n_neighbors=10 weights=uniform: MAE 0.7560
  20752.1s  knn metric=euclidean n_neighbors=10 weights=distance: MAE 0.7517
  20757.3s  knn metric=euclidean n_neighbors=20 weights=uniform: MAE 0.7608
  20762.7s  knn metric=euclidean n_neighbors=20 weights=distance: MAE 0.7568
  20768.4s  knn metric=euclidean n_neighbors=50 weights=uniform: MAE 0.7756
  20774.1s  knn metric=euclidean n_neighbors=50 weights=distance: MAE 0.7719
  20780.6s  knn metric=euclidean n_neighbors=100 weights=uniform: MAE 0.7916
  20787.1s  knn metric=euclidean n_neighbors=100 weights=distance: MAE 0.7879
  20855.8s  knn metric=manhattan n_neighbors=3 weights=uniform: MAE 0.7608
  20924.4s  knn metric=manhattan n_neighbors=3 weights=distance: MAE 0.7518
  20993.0s  knn metric=manhattan n_neighbors=5 weights=uniform: MAE 0.7525
  21061.6s  knn metric=manhattan n_neighbors=5 weights=distance: MAE 0.7428
  21130.1s  knn metric=manhattan n_neighbors=10 weights=uniform: MAE 0.7412
  21198.7s  knn metric=manhattan n_neighbors=10 weights=distance: MAE 0.7338
  21267.8s  knn metric=manhattan n_neighbors=20 weights=uniform: MAE 0.7445
  21337.1s  knn metric=manhattan n_neighbors=20 weights=distance: MAE 0.7387
  21407.3s  knn metric=manhattan n_neighbors=50 weights=uniform: MAE 0.7613
  21477.7s  knn metric=manhattan n_neighbors=50 weights=distance: MAE 0.7565
  21548.9s  knn metric=manhattan n_neighbors=100 weights=uniform: MAE 0.7775
  21620.1s  knn metric=manhattan n_neighbors=100 weights=distance: MAE 0.7730
  21621.1s  linear: MAE 0.8207
  21621.2s  median: MAE 1.0259
  21621.8s  ridge: MAE 0.8178
  21625.5s  tree max_depth=3 min_samples_leaf=1: MAE 0.8515
  21629.3s  tree max_depth=3 min_samples_leaf=5: MAE 0.8515
  21633.0s  tree max_depth=3 min_samples_leaf=20: MAE 0.8515
  21636.8s  tree max_depth=3 min_samples_leaf=100: MAE 0.8493
  21642.7s  tree max_depth=5 min_samples_leaf=1: MAE 0.8107
  21648.9s  tree max_depth=5 min_samples_leaf=5: MAE 0.8060
  21654.7s  tree max_depth=5 min_samples_leaf=20: MAE 0.8004
  21660.6s  tree max_depth=5 min_samples_leaf=100: MAE 0.7925
  21671.8s  tree max_depth=10 min_samples_leaf=1: MAE 0.8298
  21683.3s  tree max_depth=10 min_samples_leaf=5: MAE 0.8090
  21693.5s  tree max_depth=10 min_samples_leaf=20: MAE 0.7848
  21702.3s  tree max_depth=10 min_samples_leaf=100: MAE 0.7709
  21721.0s  tree max_depth=20 min_samples_leaf=1: MAE 0.9459
  21736.9s  tree max_depth=20 min_samples_leaf=5: MAE 0.8736
  21749.6s  tree max_depth=20 min_samples_leaf=20: MAE 0.7962
  21758.6s  tree max_depth=20 min_samples_leaf=100: MAE 0.7708
  21781.1s  tree max_depth=None min_samples_leaf=1: MAE 0.9600
  21798.2s  tree max_depth=None min_samples_leaf=5: MAE 0.8806
  21811.3s  tree max_depth=None min_samples_leaf=20: MAE 0.7966
  21820.2s  tree max_depth=None min_samples_leaf=100: MAE 0.7708
  21921.3s                                                  model      mae  fold_std    train_mae    mae_1    mae_2    mae_3    mae_4    mae_5
  21921.3s                              forest min_samples_leaf=3 0.684220  0.011097 3.205755e-01 1.002533 0.095986 0.377798 0.841244 1.103540
  21921.3s                             forest min_samples_leaf=10 0.686526  0.010959 4.887841e-01 1.014548 0.095836 0.383082 0.827968 1.111198
  21921.3s                              forest min_samples_leaf=1 0.691337  0.011462 2.541614e-01 1.005064 0.097914 0.376612 0.872996 1.104097
  21921.3s   knn metric=manhattan n_neighbors=10 weights=distance 0.733771  0.013115 0.000000e+00 1.093793 0.118018 0.424367 0.861181 1.171497
  21921.3s   knn metric=manhattan n_neighbors=20 weights=distance 0.738696  0.012981 0.000000e+00 1.106604 0.116763 0.422791 0.856714 1.190610
  21921.3s    knn metric=manhattan n_neighbors=10 weights=uniform 0.741216  0.013123 6.640878e-01 1.106558 0.118267 0.424762 0.864433 1.192061
  21921.3s    knn metric=manhattan n_neighbors=5 weights=distance 0.742807  0.019126 0.000000e+00 1.101647 0.121210 0.433673 0.885465 1.172039
  21921.3s    knn metric=manhattan n_neighbors=20 weights=uniform 0.744474  0.013327 7.037116e-01 1.116478 0.116961 0.424195 0.859721 1.205015
  21921.3s   knn metric=euclidean n_neighbors=10 weights=distance 0.751684  0.015615 1.541489e-07 1.130168 0.121491 0.434894 0.868036 1.203828
  21921.3s    knn metric=manhattan n_neighbors=3 weights=distance 0.751786  0.023785 0.000000e+00 1.099046 0.123894 0.448947 0.919780 1.167261
  21921.3s     knn metric=manhattan n_neighbors=5 weights=uniform 0.752514  0.019018 5.995194e-01 1.119340 0.121180 0.433924 0.890329 1.197798
  21921.3s    knn metric=euclidean n_neighbors=10 weights=uniform 0.756015  0.016127 6.820452e-01 1.137151 0.121711 0.435510 0.870911 1.214792
  21921.3s   knn metric=manhattan n_neighbors=50 weights=distance 0.756475  0.013720 0.000000e+00 1.132336 0.116203 0.433049 0.878763 1.222025
  21921.3s   knn metric=euclidean n_neighbors=20 weights=distance 0.756790  0.015874 3.034152e-07 1.138255 0.118990 0.433103 0.863463 1.230139
  21921.3s    knn metric=euclidean n_neighbors=20 weights=uniform 0.760761  0.016278 7.205286e-01 1.144501 0.119128 0.434122 0.866558 1.239494
  21921.3s     knn metric=manhattan n_neighbors=3 weights=uniform 0.760838  0.023222 5.166231e-01 1.116302 0.123489 0.449219 0.921107 1.194073
  21921.3s    knn metric=manhattan n_neighbors=50 weights=uniform 0.761288  0.013999 7.442233e-01 1.139614 0.116566 0.435107 0.883523 1.231629
  21921.3s    knn metric=euclidean n_neighbors=5 weights=distance 0.763918  0.011293 7.440622e-08 1.142931 0.124879 0.445594 0.905332 1.200855
  21921.3s     knn metric=euclidean n_neighbors=5 weights=uniform 0.769238  0.011533 6.165326e-01 1.151903 0.124867 0.445724 0.908975 1.214722
  21921.3s                 tree max_depth=20 min_samples_leaf=100 0.770808  0.015226 7.165883e-01 1.137485 0.131817 0.456718 0.913971 1.214049
  21921.3s               tree max_depth=None min_samples_leaf=100 0.770808  0.015226 7.165883e-01 1.137485 0.131817 0.456718 0.913971 1.214049
  21921.3s                 tree max_depth=10 min_samples_leaf=100 0.770917  0.014654 7.175403e-01 1.137485 0.131211 0.457869 0.913971 1.214049
  21921.3s   knn metric=euclidean n_neighbors=50 weights=distance 0.771941  0.013730 7.240673e-07 1.161403 0.117690 0.437196 0.883879 1.259538
  21921.3s  knn metric=manhattan n_neighbors=100 weights=distance 0.772962  0.013816 0.000000e+00 1.158451 0.116390 0.441700 0.900237 1.248031
  21921.3s    knn metric=euclidean n_neighbors=50 weights=uniform 0.775562  0.013812 7.580540e-01 1.166765 0.117903 0.438924 0.887949 1.266269
  21921.3s    knn metric=euclidean n_neighbors=3 weights=distance 0.775625  0.009908 4.122712e-08 1.155190 0.129734 0.458458 0.928500 1.206242
  21921.3s   knn metric=manhattan n_neighbors=100 weights=uniform 0.777539  0.014036 7.689564e-01 1.165094 0.116824 0.444193 0.905787 1.255799
  21921.3s     knn metric=euclidean n_neighbors=3 weights=uniform 0.779848  0.010097 5.387545e-01 1.163215 0.129721 0.459015 0.930927 1.216363
  21921.3s                  tree max_depth=10 min_samples_leaf=20 0.784841  0.021730 6.249650e-01 1.151356 0.120403 0.437214 0.952628 1.262605
  21921.3s  knn metric=euclidean n_neighbors=100 weights=distance 0.787851  0.015346 1.383736e-06 1.180772 0.117387 0.445829 0.910966 1.284299
  21921.3s   knn metric=euclidean n_neighbors=100 weights=uniform 0.791571  0.015637 7.831424e-01 1.185629 0.117637 0.448101 0.916546 1.289941
  21921.3s                  tree max_depth=5 min_samples_leaf=100 0.792489  0.018457 7.650732e-01 1.159878 0.151025 0.466861 0.933434 1.251246
  21921.3s                  tree max_depth=20 min_samples_leaf=20 0.796189  0.021927 5.786715e-01 1.167293 0.120607 0.436922 0.962840 1.293284
  21921.3s                tree max_depth=None min_samples_leaf=20 0.796581  0.022010 5.783401e-01 1.167293 0.122265 0.437222 0.962840 1.293284
  21921.3s                   tree max_depth=5 min_samples_leaf=20 0.800370  0.017062 7.564139e-01 1.160758 0.135438 0.471698 0.981077 1.252878
  21921.3s                    tree max_depth=5 min_samples_leaf=5 0.806014  0.018936 7.542703e-01 1.161982 0.132129 0.487205 0.992231 1.256523
  21921.3s                   tree max_depth=10 min_samples_leaf=5 0.809043  0.028339 5.480884e-01 1.167103 0.119159 0.469885 0.995436 1.293629
  21921.3s                    tree max_depth=5 min_samples_leaf=1 0.810718  0.017594 7.544618e-01 1.165661 0.135238 0.496604 0.995919 1.260167
  21921.3s                                                  ridge 0.817827  0.027541 7.883796e-01 1.082924 0.224179 0.485044 1.136185 1.160804
  21921.3s                                                 linear 0.820657  0.025228 7.896841e-01 1.071439 0.226807 0.504142 1.144159 1.156737
  21921.3s                   tree max_depth=10 min_samples_leaf=1 0.829752  0.027356 5.180530e-01 1.195532 0.128192 0.499353 1.025970 1.299713
  21921.3s                  tree max_depth=3 min_samples_leaf=100 0.849325  0.010165 8.381850e-01 1.224821 0.189898 0.514979 0.977805 1.339121
  21921.3s                    tree max_depth=3 min_samples_leaf=1 0.851453  0.011955 8.397471e-01 1.225283 0.190343 0.514732 0.977597 1.349309
  21921.3s                    tree max_depth=3 min_samples_leaf=5 0.851453  0.011955 8.397471e-01 1.225283 0.190343 0.514732 0.977597 1.349309
  21921.3s                   tree max_depth=3 min_samples_leaf=20 0.851453  0.011955 8.397471e-01 1.225283 0.190343 0.514732 0.977597 1.349309
  21921.3s                   tree max_depth=20 min_samples_leaf=5 0.873630  0.032113 3.180969e-01 1.265726 0.123586 0.472970 1.076600 1.429269
  21921.3s                 tree max_depth=None min_samples_leaf=5 0.880605  0.031726 3.068968e-01 1.279208 0.131597 0.479018 1.075018 1.438185
  21921.3s                   tree max_depth=20 min_samples_leaf=1 0.945866  0.037436 6.733979e-02 1.362367 0.126399 0.508393 1.240996 1.491176
  21921.3s                 tree max_depth=None min_samples_leaf=1 0.959987  0.041036 0.000000e+00 1.379330 0.139425 0.529893 1.230054 1.521233
  21921.3s                                                 median 1.025912  0.017878 1.025912e+00 1.662653 0.185252 0.575999 0.769050 1.936608
  21984.5s  {"rows": 25, "max_relative": 4.991129275427212e-07, "max_absolute": 0.037500000005820766, "worst_feature": "texture_gabor_w16_o0", "worst_table": 1000.799, "worst_computed": 1000.7994995117188}
  21988.1s  /usr/local/lib/python3.13/dist-packages/mistune.py:435: SyntaxWarning: invalid escape sequence '\|'
  21988.1s    cells[i][c] = re.sub('\\\\\|', '|', cell)
  21988.4s  /usr/local/lib/python3.13/dist-packages/nbconvert/filters/filter_links.py:36: SyntaxWarning: invalid escape sequence '\_'
  21988.4s    text = re.sub(r'_', '\_', text) # Escape underscores in display text
  21989.6s  [NbConvertApp] Converting notebook __script__.ipynb to html
  21990.3s  [NbConvertApp] Writing 411119 bytes to __results__.html
```

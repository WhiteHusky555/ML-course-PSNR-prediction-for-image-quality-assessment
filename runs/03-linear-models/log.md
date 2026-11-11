# Лог прогона 03. Линейные модели

Пояснения к прогону — в [README.md](README.md). Время в начале строки отсчитывается от запуска ядра.

```text
     13.3s  {"python": "3.13.18", "numpy": "2.1.3", "pandas": "2.3.3", "scikit-learn": "1.6.1", "scipy": "1.16.3", "scikit-image": "0.25.2", "opencv-python": "4.14.0.94", "opencv-python-headless": "4.14.0.94", "pyyaml": "6.0.3", "pytest": "8.4.2"}
     27.0s  ........................                                                 [100%]
     27.0s  24 passed in 9.25s
     55.6s  elasticnet alpha=0.0001 l1_ratio=0.1: MAE 0.8068
     70.6s  elasticnet alpha=0.0001 l1_ratio=0.5: MAE 0.8058
     82.4s  elasticnet alpha=0.0001 l1_ratio=0.9: MAE 0.8042
     88.7s  elasticnet alpha=0.001 l1_ratio=0.1: MAE 0.8090
     90.8s  elasticnet alpha=0.001 l1_ratio=0.5: MAE 0.8087
     91.9s  elasticnet alpha=0.001 l1_ratio=0.9: MAE 0.8071
     93.3s  elasticnet alpha=0.01 l1_ratio=0.1: MAE 0.8169
     93.7s  elasticnet alpha=0.01 l1_ratio=0.5: MAE 0.8213
     94.1s  elasticnet alpha=0.01 l1_ratio=0.9: MAE 0.8289
     94.4s  elasticnet alpha=0.1 l1_ratio=0.1: MAE 0.8465
     94.7s  elasticnet alpha=0.1 l1_ratio=0.5: MAE 0.8747
     95.1s  elasticnet alpha=0.1 l1_ratio=0.9: MAE 0.9014
    109.2s  lasso alpha=0.0001: MAE 0.8037
    115.3s  lasso alpha=0.0003: MAE 0.8031
    116.5s  lasso alpha=0.001: MAE 0.8062
    117.0s  lasso alpha=0.003: MAE 0.8147
    117.3s  lasso alpha=0.01: MAE 0.8314
    117.7s  lasso alpha=0.03: MAE 0.8571
    118.0s  lasso alpha=0.1: MAE 0.9093
    118.3s  lasso alpha=0.3: MAE 1.0163
    119.1s  linear: MAE 0.8100
    119.1s  median: MAE 1.0260
    119.5s  ridge alpha=0.01: MAE 0.8084
    120.0s  ridge alpha=0.1: MAE 0.8071
    120.6s  ridge alpha=1.0: MAE 0.8069
    121.2s  ridge alpha=10.0: MAE 0.8092
    121.8s  ridge alpha=100.0: MAE 0.8154
    122.3s  ridge alpha=1000.0: MAE 0.8338
    122.8s  ridge alpha=10000.0: MAE 0.8916
    124.2s                                 model      mae  fold_std    mae_1    mae_2    mae_3    mae_4    mae_5
    124.2s                    lasso alpha=0.0003 0.803060  0.016159 1.090727 0.217804 0.478007 1.059241 1.169518
    124.2s                    lasso alpha=0.0001 0.803678  0.017251 1.081744 0.220881 0.485232 1.067282 1.163253
    124.2s  elasticnet alpha=0.0001 l1_ratio=0.9 0.804162  0.016937 1.083818 0.221974 0.484102 1.067056 1.163862
    124.2s  elasticnet alpha=0.0001 l1_ratio=0.5 0.805811  0.017014 1.087128 0.224400 0.483101 1.069543 1.164883
    124.2s                     lasso alpha=0.001 0.806201  0.017111 1.105427 0.215922 0.481281 1.048697 1.179678
    124.2s  elasticnet alpha=0.0001 l1_ratio=0.1 0.806765  0.017181 1.088711 0.225744 0.483165 1.070895 1.165307
    124.2s                       ridge alpha=1.0 0.806939  0.017193 1.088716 0.226025 0.483459 1.071249 1.165243
    124.2s   elasticnet alpha=0.001 l1_ratio=0.9 0.807070  0.017028 1.105880 0.217015 0.482012 1.050619 1.179824
    124.2s                       ridge alpha=0.1 0.807091  0.016762 1.079694 0.226247 0.492170 1.074822 1.162520
    124.2s                      ridge alpha=0.01 0.808361  0.017144 1.077099 0.226425 0.496134 1.079638 1.162507
    124.2s   elasticnet alpha=0.001 l1_ratio=0.5 0.808660  0.017085 1.106187 0.220078 0.481867 1.055126 1.180044
    124.2s   elasticnet alpha=0.001 l1_ratio=0.1 0.809046  0.017392 1.104195 0.223833 0.481348 1.057832 1.178020
    124.2s                      ridge alpha=10.0 0.809195  0.017448 1.103341 0.224811 0.481295 1.059340 1.177189
    124.2s                                linear 0.810013  0.017793 1.076826 0.229050 0.501137 1.080486 1.162565
    124.2s                     lasso alpha=0.003 0.814730  0.016587 1.121456 0.226016 0.500067 1.030918 1.195191
    124.2s                     ridge alpha=100.0 0.815379  0.015562 1.125682 0.220627 0.495411 1.032992 1.202185
    124.2s    elasticnet alpha=0.01 l1_ratio=0.1 0.816949  0.015477 1.129338 0.223918 0.498978 1.026771 1.205738
    124.2s    elasticnet alpha=0.01 l1_ratio=0.5 0.821264  0.016008 1.138910 0.228499 0.512137 1.011213 1.215559
    124.2s    elasticnet alpha=0.01 l1_ratio=0.9 0.828874  0.015091 1.149950 0.233241 0.526418 1.007777 1.226984
    124.2s                      lasso alpha=0.01 0.831358  0.014658 1.153861 0.234902 0.529944 1.007744 1.230337
    124.2s                    ridge alpha=1000.0 0.833758  0.013352 1.163571 0.209209 0.516719 1.027752 1.251538
    124.2s     elasticnet alpha=0.1 l1_ratio=0.1 0.846455  0.011668 1.187590 0.212340 0.536996 1.020865 1.274484
    124.2s                      lasso alpha=0.03 0.857144  0.012401 1.191941 0.228937 0.558005 1.033777 1.273060
    124.2s     elasticnet alpha=0.1 l1_ratio=0.5 0.874665  0.010222 1.231216 0.227952 0.547428 1.049609 1.317121
    124.2s                   ridge alpha=10000.0 0.891648  0.008960 1.285591 0.177449 0.518600 1.112773 1.363826
    124.2s     elasticnet alpha=0.1 l1_ratio=0.9 0.901409  0.008118 1.266078 0.253051 0.529840 1.094341 1.363736
    124.2s                       lasso alpha=0.1 0.909313  0.007727 1.277437 0.258790 0.527517 1.106785 1.376037
    124.2s                       lasso alpha=0.3 1.016260  0.007914 1.326185 0.474285 0.566984 1.249922 1.463925
    124.2s                                median 1.025969  0.018641 1.662687 0.185335 0.576075 0.769100 1.936646
    144.5s  {"rows": 25, "max_relative": 4.991129275427212e-07, "max_absolute": 0.037500000005820766, "worst_feature": "texture_gabor_w16_o0", "worst_table": 1000.799, "worst_computed": 1000.7994995117188}
    147.4s  /usr/local/lib/python3.13/dist-packages/mistune.py:435: SyntaxWarning: invalid escape sequence '\|'
    147.4s    cells[i][c] = re.sub('\\\\\|', '|', cell)
    147.8s  /usr/local/lib/python3.13/dist-packages/nbconvert/filters/filter_links.py:36: SyntaxWarning: invalid escape sequence '\_'
    147.8s    text = re.sub(r'_', '\_', text) # Escape underscores in display text
    148.8s  [NbConvertApp] Converting notebook __script__.ipynb to html
    150.2s  [NbConvertApp] Writing 356786 bytes to __results__.html
```

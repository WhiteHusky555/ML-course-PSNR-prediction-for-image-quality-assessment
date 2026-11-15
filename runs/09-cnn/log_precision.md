# Лог прогона 09. CNN: сравнение float32 и смешанной точности

Пояснения к прогону — в [README.md](README.md). Время в начале строки отсчитывается от запуска ядра.

```text
      3.6s  3406e29d2070877cb8e36046b00950d45d5d6a55 Сравнение обучения CNN во float32 и со смешанной точностью
      3.6s  
     51.2s  ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
     51.2s  bigframes 2.51.0 requires google-cloud-bigquery-storage<3.0.0,>=2.30.0, which is not installed.
     51.2s  ydata-profiling 4.18.4 requires matplotlib<=3.10,>=3.5, but you have matplotlib 3.10.9 which is incompatible.
     51.5s  3.13.16 Tesla T4, 15360 MiB
     51.5s  Tesla T4, 15360 MiB
     51.5s  
    147.8s  {'backbone': 'small', 'input': 'full', 'batch_size': 16, 'precision': 'float32', 'step_seconds': 0.852180946499999, 'images_per_second': 18.775355240825043, 'peak_memory_mb': 9753.192626953125}
    182.6s  {'backbone': 'small', 'input': 'full', 'batch_size': 16, 'precision': 'mixed_float16', 'step_seconds': 0.666954753500022, 'images_per_second': 23.98963335374065, 'peak_memory_mb': 4795.245361328125}
    206.3s  {'backbone': 'small', 'input': 'crop', 'batch_size': 32, 'precision': 'float32', 'step_seconds': 0.37670027499999037, 'images_per_second': 84.94817265530486, 'peak_memory_mb': 5166.5205078125}
    221.9s  {'backbone': 'small', 'input': 'crop', 'batch_size': 32, 'precision': 'mixed_float16', 'step_seconds': 0.2684919954999998, 'images_per_second': 119.18418625630879, 'peak_memory_mb': 2013.337890625}
    223.2s  Downloading data from https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5
    223.3s  16705208/16705208 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step
    235.1s  WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    235.1s  E0000 00:00:1794663257.735353      47 meta_optimizer.cc:967] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape insequential_1/efficientnetb0_1/block2b_drop_1/stateless_dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
    275.9s  {'backbone': 'efficientnetb0', 'input': 'full', 'batch_size': 16, 'precision': 'float32', 'step_seconds': 0.9946962219999875, 'images_per_second': 16.085312928835272, 'peak_memory_mb': 8705.18603515625}
    359.2s  {'backbone': 'efficientnetb0', 'input': 'full', 'batch_size': 16, 'precision': 'mixed_float16', 'step_seconds': 0.7732103074999372, 'images_per_second': 20.69294711258269, 'peak_memory_mb': 6100.87646484375}
    372.7s  E0000 00:00:1794663395.324256      47 meta_optimizer.cc:967] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape insequential_1/efficientnetb0_1/block2b_drop_1/stateless_dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
    391.3s  {'backbone': 'efficientnetb0', 'input': 'crop', 'batch_size': 32, 'precision': 'float32', 'step_seconds': 0.42903328399995644, 'images_per_second': 74.58628780885739, 'peak_memory_mb': 3906.052978515625}
    442.1s  {'backbone': 'efficientnetb0', 'input': 'crop', 'batch_size': 32, 'precision': 'mixed_float16', 'step_seconds': 0.3537523985000348, 'images_per_second': 90.45875062808048, 'peak_memory_mb': 2842.628662109375}
    507.6s        backbone input  batch_size     precision  step_seconds  images_per_second  peak_memory_mb  speedup
    507.6s           small  full          16       float32      0.852181          18.775355     9753.192627 1.000000
    507.6s           small  full          16 mixed_float16      0.666955          23.989633     4795.245361 1.277719
    507.6s           small  crop          32       float32      0.376700          84.948173     5166.520508 1.000000
    507.6s           small  crop          32 mixed_float16      0.268492         119.184186     2013.337891 1.403022
    507.6s  efficientnetb0  full          16       float32      0.994696          16.085313     8705.186035 1.000000
    507.6s  efficientnetb0  full          16 mixed_float16      0.773210          20.692947     6100.876465 1.286450
    507.6s  efficientnetb0  crop          32       float32      0.429033          74.586288     3906.052979 1.000000
    507.6s  efficientnetb0  crop          32 mixed_float16      0.353752          90.458751     2842.628662 1.212807
    507.6s  
    507.6s        backbone  mean_gap  max_gap
    507.6s           small  0.010417 0.025153
    507.6s  efficientnetb0  0.002434 0.007999
    519.0s  /usr/local/lib/python3.13/dist-packages/mistune.py:435: SyntaxWarning: invalid escape sequence '\|'
    519.0s    cells[i][c] = re.sub('\\\\\|', '|', cell)
    519.4s  /usr/local/lib/python3.13/dist-packages/nbconvert/filters/filter_links.py:36: SyntaxWarning: invalid escape sequence '\_'
    519.4s    text = re.sub(r'_', '\_', text) # Escape underscores in display text
    521.2s  [NbConvertApp] Converting notebook __script__.ipynb to html
    521.8s  [NbConvertApp] Writing 278993 bytes to __results__.html
```

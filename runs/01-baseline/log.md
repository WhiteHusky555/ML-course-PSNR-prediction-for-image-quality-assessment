# Лог прогона 01. Каркас пайплайна и медиана

Пояснения к прогону — в [README.md](README.md). Время в начале строки отсчитывается от запуска ядра.

```text
     14.2s  {"python": "3.13.18", "numpy": "2.1.3", "pandas": "2.3.3", "scikit-learn": "1.6.1", "scipy": "1.16.3", "scikit-image": "0.25.2", "opencv-python": "4.14.0.94", "opencv-python-headless": "4.14.0.94", "pyyaml": "6.0.3", "pytest": "8.4.2"}
     26.4s  ..................                                                       [100%]
     26.4s  18 passed in 7.96s
     30.3s   model      mae  fold_std    mae_1    mae_2    mae_3  mae_4    mae_5
     30.3s  median 1.025969  0.018641 1.662687 0.185335 0.576075 0.7691 1.936646
     57.7s  {"rows": 25, "max_relative": 4.991129275427212e-07, "max_absolute": 0.037500000005820766, "worst_feature": "texture_gabor_w16_o0", "worst_table": 1000.799, "worst_computed": 1000.7994995117188}
     61.1s  /usr/local/lib/python3.13/dist-packages/mistune.py:435: SyntaxWarning: invalid escape sequence '\|'
     61.1s    cells[i][c] = re.sub('\\\\\|', '|', cell)
     61.5s  /usr/local/lib/python3.13/dist-packages/nbconvert/filters/filter_links.py:36: SyntaxWarning: invalid escape sequence '\_'
     61.5s    text = re.sub(r'_', '\_', text) # Escape underscores in display text
     62.6s  [NbConvertApp] Converting notebook __script__.ipynb to html
     64.0s  [NbConvertApp] Writing 342496 bytes to __results__.html
```

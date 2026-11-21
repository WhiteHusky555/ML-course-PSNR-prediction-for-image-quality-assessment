# Прогоны на Kaggle

Прогоны 01–12 выполнялись в приватных ядрах Kaggle с подключёнными данными соревнования: прогоны 01–08 и 10 на CPU (4 ядра, Python 3.13), прогон 09 со свёрточными сетями на GPU Tesla T4. В этой папке собраны логи успешно завершённых прогонов: каждый прогон лежит в отдельной папке с логом ядра (`log.md`), таблицами результатов и пояснениями (`README.md`). Время в начале строк лога отсчитывается от запуска ядра.

Проверочные прогоны 01 и 03–06 запускают код репозитория в состоянии очередного этапа: записывает версии библиотек, запускает тесты, затем `main.py` с основным конфигом и с каждой конфигурацией из `experiments/` и в конце пересчитывает признаки для 25 изображений и сравнивает их с таблицей `features/features.csv.gz`. Таблицы `results_<конфигурация>.csv` — это файлы `outputs/.../results.csv`, которые пайплайн сохранил в этом прогоне. В столбце «Код» указан коммит, на коде которого получен лог; ссылки на файлы приведены в README прогона.

| Папка | Что проверялось | Тесты | Время | Итог | Код |
|---|---|---|---|---|---|
| [01-baseline](01-baseline) | каркас пайплайна, медиана | 18 | 1 мин | MAE 1,026 | [86cec71](https://github.com/WhiteHusky555/ML-course-PSNR-prediction-for-image-quality-assessment/commit/86cec71) |
| [02-eda](02-eda) | выполнение `eda.ipynb` | — | 5 мин | ноутбук с результатами | [001fb64](https://github.com/WhiteHusky555/ML-course-PSNR-prediction-for-image-quality-assessment/commit/001fb64) |
| [03-linear-models](03-linear-models) | линейные модели | 24 | 2,5 мин | Lasso 0,803 | [52a611b](https://github.com/WhiteHusky555/ML-course-PSNR-prediction-for-image-quality-assessment/commit/52a611b) |
| [04-preprocessing](04-preprocessing) | исследования предобработки, кодирования и валидации, ограничение квантилями | 36 | 5 мин | Lasso 0,800 | [6b79cee](https://github.com/WhiteHusky555/ML-course-PSNR-prediction-for-image-quality-assessment/commit/6b79cee) |
| [05-knn-tree-forest](05-knn-tree-forest) | KNN, решающее дерево, случайный лес | 40 | 2,5 ч | Random Forest 0,688 | [cf34382](https://github.com/WhiteHusky555/ML-course-PSNR-prediction-for-image-quality-assessment/commit/cf34382) |
| [06-boosting](06-boosting) | градиентный бустинг, пять реализаций | 48 | 6,1 ч | XGBoost 0,646 | [f3893f8](https://github.com/WhiteHusky555/ML-course-PSNR-prediction-for-image-quality-assessment/commit/f3893f8) |
| [07-tuning](07-tuning) | подбор гиперпараметров LightGBM, XGBoost и CatBoost (Optuna) | 52 | 9,1 ч | LightGBM 0,645 на контрольных разбиениях | [feb9dbd](https://github.com/WhiteHusky555/ML-course-PSNR-prediction-for-image-quality-assessment/commit/feb9dbd) |
| [08-mlp](08-mlp) | полносвязная нейросеть на Keras: 35 сетей по пунктам чек-листа | 60 | 1,5 ч | MLP 0,652 | [bd8538b](https://github.com/WhiteHusky555/ML-course-PSNR-prediction-for-image-quality-assessment/commit/bd8538b) |
| [09-cnn](09-cnn) | свёрточные сети на изображениях: сеть с нуля и EfficientNetB0 на целых изображениях и на вырезках, сравнение float32 и смешанной точности; пять прогонов на GPU, логи в отдельных файлах | 69 | 6,6 ч | EfficientNetB0 0,493 | [bacbf68](https://github.com/WhiteHusky555/ML-course-PSNR-prediction-for-image-quality-assessment/commit/bacbf68) |
| [10-ensemble](10-ensemble) | ансамбль из прогнозов моделей на признаках и свёрточных сетей: среднее, взвешенное среднее, стекинг | 81 | 21 мин | стекинг на прогнозах 0,461 | [94f1df6](https://github.com/WhiteHusky555/ML-course-PSNR-prediction-for-image-quality-assessment/commit/94f1df6) |

Во всех проверочных прогонах сверка признаков дала одинаковый результат: наибольшее относительное расхождение 5·10⁻⁷. Код `src/features.py` воспроизводит таблицу признаков с точностью до округления до 7 значащих цифр.

Предупреждения `SyntaxWarning` от `mistune` и `nbconvert`, сообщения `NbConvertApp` и `PyparsingDeprecationWarning` от `matplotlib` выводят библиотеки образа Kaggle при сохранении результатов ядра. К коду проекта они не относятся.

## Локальное дополнение ансамблей

[13-linear-ensemble](13-linear-ensemble) — сравнение LinearRegression и Ridge с прежними ансамблями на сохранённых прогнозах для 155 и 167 признаков. Расчёт на CPU Windows занял 53 секунды, 33 профильных теста прошли. Лучший Ridge в проверенной сетке: MAE 0,505995; LightGBM-стекинг остался лучше (0,450888 на 155 признаках).

Сравнительные сабмиты Ridge alpha = 1000: на 155 признаках public/private MAE 0,46667/0,47736 (№ 58462179), на 167 — 0,46523/0,47504 (№ 58462231). Оба уступили LightGBM-стекингу прогона 11 (0,41641/0,41627). CSV и ответы Kaggle сохранены в `13-linear-ensemble/submissions/`.

[14-voting](14-voting) — локальный эксперимент VotingRegressor на 155 признаках: Ridge, дерево, HistGradientBoosting, пять групповых фолдов. MAE 0,687948 против 0,660950 у лучшего участника. Прогнозы совпали с ручным средним; результаты и проверка сохранены.

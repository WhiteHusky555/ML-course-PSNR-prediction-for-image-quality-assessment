# Прогоны на Kaggle

Все расчёты проекта выполнялись в приватных ядрах Kaggle на CPU (4 ядра, Python 3.13), к которым подключены данные соревнования. В этой папке собраны логи успешно завершённых прогонов: каждый прогон лежит в отдельной папке с логом ядра (`log.md`), таблицами результатов и пояснениями (`README.md`). Время в начале строк лога отсчитывается от запуска ядра.

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

Во всех проверочных прогонах сверка признаков дала одинаковый результат: наибольшее относительное расхождение 5·10⁻⁷. Код `src/features.py` воспроизводит таблицу признаков с точностью до округления до 7 значащих цифр.

Предупреждения `SyntaxWarning` от `mistune` и `nbconvert`, сообщения `NbConvertApp` и `PyparsingDeprecationWarning` от `matplotlib` выводят библиотеки образа Kaggle при сохранении результатов ядра. К коду проекта они не относятся.

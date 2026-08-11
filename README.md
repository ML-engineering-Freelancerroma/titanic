# Kaggle-соревнование о предсказании выживших на корабле "Titanic"

Этот репозиторий содержит решение задачи **"Titanic: Machine Learning from Disaster"** – бинарной классификации пассажиров.

**Датасет:** [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data)

---

## Структура репозитория

### 1. `feat_prepare.ipynb`
Jupyter Notebook для **предобработки и разведочного анализа данных (EDA)**.

**Основные этапы:**
- Загрузка исходного `train.csv`
- Заполнение пропусков в `Age` по медиане в группах `(Pclass, Sex)`
- Извлечение палубы (`Deck`) из `Cabin` и подсчёт количества кают (`Cab_count`)
- Заполнение пропусков в `Embarked` по модальному порту в группе `(Pclass, Fare_round)`
- Кодирование `Sex` (Label Encoding) и One‑Hot кодирование `Embarked`
- Таргет‑энкодинг `Deck` с кросс‑валидацией (KFold)
- Извлечение обращения из `Name` → `Designation`, объединение редких обращений в `Uniq`
- Создание признака `Family = SibSp + Parch`
- Создание бинарного признака `Ticket_uniq` – уникальность билета
- Визуализация распределений, корреляций, t‑SNE
- Сохранение очищенного датасета в `train_prep.csv`

**Результат:** `train_prep.csv` – подготовленный набор для обучения моделей.

### 2. `core.py`
Модуль с основными утилитами, которые используются во всех частях проекта.

**Функции:**

- `evaluate_model()` – кросс-валидация модели по нескольким метрикам
- `save_results()` – сохранение результатов в CSV
- `save_model()` – сохранение модели через `joblib`
- `fit_preprocessor()` – обучение пайплайна предобработки на тренировочных данных (возвращает `X_train`, `y_train`, словарь `params`)
- `transform_preprocessor()` – применение обученного пайплайна к тестовым данным
- `evaluate_models_on_test()` – загрузка нескольких моделей, предсказание и сравнение метрик на тестовом наборе

### 3. `train_CatBoost.ipynb`
Jupyter Notebook для обучения **CatBoostClassifier** с подбором гиперпараметров.

**Основные этапы:**
- Импорт `X_train`, `Y_train`, `skf`, `CAT_FEATURES` из `core.py`
- Создание базового классификатора с `auto_class_weights='Balanced'` и ранней остановкой (`od_type='Iter'`, `od_wait=50`)
- Определение сетки гиперпараметров: `iterations`, `depth`, `learning_rate`, `l2_leaf_reg`, `bagging_temperature`, `random_strength`, `border_count`, `grow_policy`, `min_data_in_leaf`
- Запуск `RandomizedSearchCV` (50 итераций, 5‑fold CV, метрика `roc_auc`)
- Вывод лучших параметров и ROC‑AUC
- Оценка лучшей модели на кросс‑валидации по всем метрикам (Accuracy, F1, ROC‑AUC, Precision)
- Сохранение результатов в `results_all.csv` через `save_results()`
- Сохранение модели в `models/CatBoost_best.pkl` через `save_model()`

**Результат:** обученная модель CatBoost с оптимальными гиперпараметрами и файл с метриками.

### 4. `train_MLP.ipynb`
Jupyter Notebook для обучения **MLPClassifier** с подбором гиперпараметров.

**Основные этапы:**
- Импорт `X_train`, `Y_train`, `skf` из `core.py`
- Создание пайплайна `StandardScaler` + `MLPClassifier` с ранней остановкой (`early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=10`)
- Определение сетки гиперпараметров: `hidden_layer_sizes`, `activation`, `alpha`, `learning_rate`, `learning_rate_init`
- Запуск `RandomizedSearchCV` (30 итераций, 5‑fold CV, метрика `roc_auc`)
- Вывод лучших параметров и ROC‑AUC
- Оценка лучшей модели на кросс‑валидации по всем метрикам (Accuracy, F1, ROC‑AUC, Precision)
- Сохранение результатов в `results_all.csv` через `save_results()`
- Сохранение модели в `models/MLP_best.pkl` через `save_model()`

**Результат:** обученная нейросетевая модель с оптимальными параметрами и файл с метриками.

### 5. `train_XGB.ipynb`
Jupyter Notebook для обучения **XGBoost** с подбором гиперпараметров.

**Основные этапы:**
- Импорт `X_train`, `Y_train`, `skf` из `core.py`
- Создание `XGBClassifier` с учётом дисбаланса классов (`scale_pos_weight`) и быстрым `tree_method='hist'`
- Определение сетки гиперпараметров: `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `gamma`, `reg_lambda`, `reg_alpha`, `subsample`, `colsample_bytree`
- Запуск `RandomizedSearchCV` (200 итераций, 5‑fold CV, метрика `roc_auc`)
- Вывод лучших параметров и ROC‑AUC
- Оценка лучшей модели на кросс‑валидации по всем метрикам (Accuracy, F1, ROC‑AUC, Precision)
- Сохранение результатов в `results_all.csv` через `save_results()`
- Сохранение модели в `models/XGBoost_best.pkl` через `save_model()`

**Результат:** обученная модель XGBoost с оптимальными гиперпараметрами и файл с метриками.

### 6. `test.ipynb`
Jupyter Notebook для **тестирования обученных моделей** на тестовых данных и сравнения их качества.

**Основные этапы:**
- Загрузка сырого `train.csv`, обучение препроцессора через `fit_preprocessor()` и сохранение параметров в `preprocess_params.pkl`
- Применение `transform_preprocessor()` к `test.csv` (тестовые данные)
- Загрузка сохранённых моделей из папки `models/` (CatBoost, XGBoost, MLP)
- Вычисление метрик (Accuracy, F1, Precision, Recall) и матриц ошибок на тестовых данных (используется `gender_submission.csv` как истинные ответы)
- Сравнение моделей в сводной таблице, сохранение в `test_models_comparison.csv`
- Создание файлов для отправки на Kaggle: `submission_CatBoost.csv`, `submission_XGBoost.csv`, `submission_MLP.csv`

**Результат:** файлы с предсказаниями для каждой модели и таблица сравнения метрик на тестовом наборе.
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

**Выход:** `train_prep.csv` – подготовленный набор для обучения моделей.
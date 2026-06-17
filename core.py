import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score, precision_score
import os
from datetime import datetime


DF_TRAIN = 'train_prep.csv'
DF_TEST = 'test.csv'
DF_SUB = 'gender_submission.csv'

df_train = pd.read_csv(DF_TRAIN)
X_train = df_train.drop('Survived', axis=1)
Y_train = df_train['Survived']

df_test = pd.read_csv(DF_TEST)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

CAT_FEATURES = ['Pclass', 'Sex', 'Designation']

# Словарь метрик для оценки модели
score = {
    'Accuracy': 'accuracy',
    'F1': make_scorer(f1_score, average='binary'),
    'ROC-AUC': 'roc_auc',
    'Precision': make_scorer(precision_score, average='binary'),
}


def evaluate_model(
    model,
    X,
    y,
    cv,
    scoring_dict,
    n_jobs=-1
):
    """Универсальная функция для cross-validation"""

    results = {}
    for metric_name, scorer in scoring_dict.items():
        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring=scorer,
            n_jobs=n_jobs
        )
        results[metric_name] = float(np.mean(scores))
    return results


def save_results(
    results_dict: dict,
    model_name: str,
    append: bool = True
):
    """Сохраняет результаты в CSV файл"""

    filepath = 'results_all.csv'
    df_new = pd.DataFrame([results_dict])
    df_new.insert(0, 'Model', model_name)
    metric_cols = ['Accuracy', 'F1', 'ROC-AUC', 'Precision', 'Recall']

    for col in metric_cols:
        if col in df_new.columns:
            df_new[col] = df_new[col].round(4)

    df_new['Saved_Time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if append and os.path.exists(filepath):
        df_existing = pd.read_csv(filepath)

        existing_cols = df_existing.columns.tolist()
        new_cols = df_new.columns.tolist()

        all_cols = list(dict.fromkeys(['Model'] + existing_cols + new_cols))

        df_existing = df_existing.reindex(columns=all_cols)
        df_new = df_new.reindex(columns=all_cols)

        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(filepath, index=False)
    print(f'Результаты {model_name} сохранены')


def save_model(
    model, model_name: str,
    models_dir: str = 'models'
):
    """Сохранение модели"""

    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(model, f'{models_dir}/{model_name}_best.pkl')

    print(f'Модель {model_name} сохранена')


def fit_preprocessor(
    df,
    target_col='Survived'
):
    """Обучение препроцессора на тренировочных данных"""

    df = df.copy()
    y = df[target_col].copy()

    age_medians = df.groupby(['Pclass', 'Sex'])['Age'].median().to_dict()
    df['Age'] = df.apply(
        lambda row: row['Age'] if pd.notnull(row['Age']) 
        else age_medians.get((row['Pclass'], row['Sex']), df['Age'].median()),
        axis=1
    )

    df['Deck'] = df['Cabin'].str.extract(r'^([A-Za-z])', expand=False).fillna('M')

    df['Cab_count'] = df['Cabin'].str.count(r'[A-Z]\d+').fillna(0).astype(int)
    df.drop(['Cabin'], axis=1, inplace=True)

    df['Fare_round'] = df['Fare'].round(-1)
    embarked_modes = df.groupby(['Pclass', 'Fare_round'])['Embarked'].agg(
        lambda x: x.mode()[0] if not x.mode().empty else 'S'
    ).to_dict()
    df['Embarked'] = df.apply(
        lambda row: row['Embarked'] if pd.notnull(row['Embarked'])
        else embarked_modes.get((row['Pclass'], row['Fare_round']), 'S'),
        axis=1
    )
    df.drop(['Fare_round'], axis=1, inplace=True)

    sex_mapping = {'male': 1, 'female': 0}
    df['Sex'] = df['Sex'].map(sex_mapping)

    df = pd.get_dummies(df, columns=['Embarked'], prefix='Emb', drop_first=True, dtype=int)
    embarked_columns = [col for col in df.columns if col.startswith('Emb_')]
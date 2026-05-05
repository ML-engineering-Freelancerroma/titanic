import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score, precision_score


DF_TRAIN = 'train_prep.csv'
DF_TEST = 'test.csv'
DF_SUB = 'gender_submission.csv'

df_train = pd.read_csv(DF_TRAIN)
X_train = df_train.drop('Survived', axis=1)
Y_train = df_train['Survived']

df_test = pd.read_csv(DF_TEST)
# Y_test = pd.read_csv(DF_SUB)['Survived'].values()


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

CAT_FEATURES = ['Pclass', 'Sex', 'Designation']


def evaluate_model(model, X, y, cv, scoring_dict, n_jobs=-1):
    """
    Универсальная функция для cross-validation
    """

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


"""
Словарь метрик для оценки модели
"""
score = {
    'Accuracy': 'accuracy',
    'F1': make_scorer(f1_score, average='binary'),
    'ROC-AUC': 'roc_auc',
    'Precision': make_scorer(precision_score, average='binary'),
}


def save_results(results_dict: dict, model_name: str):
    """
    Функция для сохранения результатов модели в файл
    """
    filepath = 'results_all.csv'

    df_new = pd.DataFrame([results_dict])
    df_new.insert(0, 'Model', model_name)

    for col in ['Accuracy', 'F1', 'ROC-AUC']:
        if col in df_new.columns:
            df_new[col] = df_new[col].round(4)

    df_new['Saved_Time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


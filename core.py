import pandas as pd
import numpy as np
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

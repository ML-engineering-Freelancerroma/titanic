import pandas as pd
from sklearn.model_selection import StratifiedKFold



DF_TRAIN = 'train.csv'
DF_TEST = 'test.csv'

df_train = pd.read_csv(DF_TRAIN)
X_train = df_train.drop('Survived', axis=1)
Y_train = df_train['Survived']

df_test = pd.read_csv(DF_TEST)
Y_test = pd.read_csv('gender_submission.csv')['Survived'].values()


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


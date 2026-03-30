import pandas as pd


df_train = pd.read_csv('train.csv')
X_train = df_train.drop('Survived', axis=1)
Y_train = df_train['Survived']

df_test = pd.read_csv('test.csv')
Y_test = pd.read_csv('gender_submission.csv')['Survived'].values()

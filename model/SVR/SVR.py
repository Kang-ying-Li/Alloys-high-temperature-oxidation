import matplotlib
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import joblib
import matplotlib.ticker as ticker

data = pd.read_csv('../../data/data.csv')

input_features = ['Ti', 'Al', 'Sn', 'Zr', 'Mo', 'Si', 'C', 'Nb', 'Hf', 'Ta', 'Cr', 'B', 'Mn',
                  'V', 'Co', 'Fe', 'Ni', 'W', 'Cu', 'Zn', 'Mg', 'Gd', 'Y', 'P', 'Ce', 'N', 'Er',
                  'Ca', 'Ge', 'Re', 'O', 'Ga', 'T']
output_feature = ['lnkp']

X = data[input_features].values
y = data[output_feature].values.ravel()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=4037)

X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
y_train = np.nan_to_num(y_train, nan=0.0, posinf=1e6, neginf=-1e6)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
y_test = np.nan_to_num(y_test, nan=0.0, posinf=1e6, neginf=-1e6)

def objective(trial):
    c = trial.suggest_loguniform('C', 1e-5, 1e5)
    epsilon = trial.suggest_loguniform('epsilon', 1e-5, 1e1)
    kernel = trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly', 'sigmoid'])
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])

    model = SVR(C=c, epsilon=epsilon, kernel=kernel, gamma=gamma)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    val_mse = mean_squared_error(y_test, y_pred)
    return val_mse

db_path = 'SVR_optuna_study.db'
if os.path.exists(db_path):
    study = optuna.create_study(
        study_name='SVR_tuning',
        storage=f'sqlite:///{db_path}',
        load_if_exists=True,
        direction='minimize'
    )
else:
    study = optuna.create_study(
        study_name='SVR_tuning',
        storage=f'sqlite:///{db_path}',
        direction='minimize'
    )

study.optimize(objective, n_trials=500)

best_params = study.best_params
print('Best hyperparameters:', best_params)

model = SVR(C=best_params['C'],
            epsilon=best_params['epsilon'],
            kernel=best_params['kernel'],
            gamma=best_params['gamma'])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Test MSE: {test_mse:.4f}')
print(f'Test R²: {r2:.4f}')

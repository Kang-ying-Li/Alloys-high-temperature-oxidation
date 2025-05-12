import matplotlib
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import joblib  # 用于保存模型

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
    n_neighbors = trial.suggest_int('n_neighbors', 1, 200)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    p = trial.suggest_int('p', 1, 2)  # 1 for Manhattan, 2 for Euclidean

    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    val_loss = mean_squared_error(y_test, y_pred)  # 使用MSE作为损失函数
    return val_loss

db_path = 'KNN_optuna_study.db'
if os.path.exists(db_path):
    study = optuna.create_study(
        study_name='KNN_tuning',
        storage=f'sqlite:///{db_path}',
        load_if_exists=True,
        direction='minimize'
    )
else:
    study = optuna.create_study(
        study_name='KNN_tuning',
        storage=f'sqlite:///{db_path}',
        direction='minimize'
    )

study.optimize(objective, n_trials=500)

best_params = study.best_params
print('Best hyperparameters:', best_params)

model = KNeighborsRegressor(**best_params)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Test MSE: {test_mse:.4f}')
print(f'Test R²: {r2:.4f}')

import matplotlib
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
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
    n_estimators = trial.suggest_int('n_estimators', 100, 5000)
    max_depth_option = trial.suggest_categorical('max_depth', ['None', 'Value'])
    if max_depth_option == 'None':
        max_depth_param = None
    else:
        max_depth_param = trial.suggest_int('max_depth_value', 10, 100)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])

    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth_param,
                                  min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                  max_features=max_features, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    val_mse = mean_squared_error(y_test, y_pred)
    return val_mse

db_path = 'RF_optuna_study.db'
if os.path.exists(db_path):
    study = optuna.create_study(
        study_name='RF_tuning',
        storage=f'sqlite:///{db_path}',
        load_if_exists=True,
        direction='minimize'
    )
else:
    study = optuna.create_study(
        study_name='RF_tuning',
        storage=f'sqlite:///{db_path}',
        direction='minimize'
    )

study.optimize(objective, n_trials=500)

best_params = study.best_params
print('Best hyperparameters:', best_params)

if best_params['max_depth'] == 'None':
    max_depth_param = None
else:
    max_depth_param = best_params['max_depth_value']

model = RandomForestRegressor(n_estimators=best_params['n_estimators'],
                              max_depth=max_depth_param,
                              min_samples_split=best_params['min_samples_split'],
                              min_samples_leaf=best_params['min_samples_leaf'],
                              max_features=best_params['max_features'],
                              random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Test MSE: {mse:.4f}')
print(f'Test R²: {r2:.4f}')

model_filename = 'RF_best_model.pkl'
joblib.dump(model, model_filename)
print(f'Model saved to {model_filename}')
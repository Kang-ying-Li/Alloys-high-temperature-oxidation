import matplotlib
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('../../data/data.csv')

input_features = ['Ti', 'Al', 'Sn', 'Zr', 'Mo', 'Si', 'C', 'Nb', 'Hf', 'Ta', 'Cr', 'B', 'Mn',
                  'V', 'Co', 'Fe', 'Ni', 'W', 'Cu', 'Zn', 'Mg', 'Gd', 'Y', 'P', 'Ce', 'N', 'Er',
                  'Ca', 'Ge', 'Re', 'O', 'Ga', 'T']
output_feature = ['lnkp']

X = data[input_features].values
y = data[output_feature].values.ravel()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.1, random_state=4037
)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'seed': 42
    }

    model = xgb.XGBRegressor(**params)

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    val_loss = mean_squared_error(y_test, y_pred)
    return val_loss

db_path = 'GB_optuna_study.db'
if os.path.exists(db_path):
    study = optuna.create_study(
        study_name='GB_tuning',
        storage=f'sqlite:///{db_path}',
        load_if_exists=True,
        direction='minimize'
    )
else:
    study = optuna.create_study(
        study_name='GB_tuning',
        storage=f'sqlite:///{db_path}',
        direction='minimize'
    )

study.optimize(objective, n_trials=500)

best_params = study.best_params
print('Best hyperparameters:', best_params)

final_model = xgb.XGBRegressor(**best_params, seed=42)

final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Test MSE Loss: {mse:.4f}')
print(f'Test R² Score: {r2:.4f}')


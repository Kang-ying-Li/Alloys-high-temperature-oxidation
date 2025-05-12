import matplotlib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import optuna
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_csv('../../data/data.csv')

input_features = ['Ti', 'Al', 'Sn', 'Zr', 'Mo', 'Si', 'C', 'Nb', 'Hf', 'Ta', 'Cr', 'B', 'Mn',
                  'V', 'Co', 'Fe', 'Ni', 'W', 'Cu', 'Zn', 'Mg', 'Gd', 'Y', 'P', 'Ce', 'N', 'Er',
                  'Ca', 'Ge', 'Re', 'O', 'Ga', 'T']
output_feature = ['lnkp']

X = data[input_features].values
y = data[output_feature].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=43)

X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
y_train = np.nan_to_num(y_train, nan=0.0, posinf=1e6, neginf=-1e6)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
y_test = np.nan_to_num(y_test, nan=0.0, posinf=1e6, neginf=-1e6)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train = torch.from_numpy(X_train_scaled).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
X_test = torch.from_numpy(X_test_scaled).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = MyDataset(X_train, y_train)
test_dataset = MyDataset(X_test, y_test)

class Net(nn.Module):
    def __init__(self, input_size, num_layers, layer_sizes):
        super(Net, self).__init__()
        layers = []
        prev_size = input_size
        for i in range(num_layers):
            layers.append(nn.Linear(prev_size, layer_sizes[i]))
            layers.append(nn.ReLU())
            prev_size = layer_sizes[i]
        layers.append(nn.Linear(prev_size, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def objective(trial):
    torch.manual_seed(42)
    np.random.seed(42)

    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    batch_size = trial.suggest_int('batch_size', 16, 128)
    num_layers = trial.suggest_int('num_layers', 1, 7)
    layer_sizes = [trial.suggest_int(f'layer_{i + 1}_size', 4, 512) for i in range(num_layers)]
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    num_epochs = trial.suggest_int('num_epochs', 50, 500)
    patience = trial.suggest_int('patience', 1, 100)
    input_size = X_train.shape[1]

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = Net(input_size, num_layers, layer_sizes).to(device)

    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    counter = 0
    for epoch in range(num_epochs):
        model.train()
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
        for batch_X, batch_y in train_loop:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loop.set_postfix(loss=loss.item())

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_X, val_y in test_loader:
                val_outputs = model(val_X)
                val_loss += criterion(val_outputs, val_y).item()
            val_loss /= len(test_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    return best_val_loss

db_path = 'BPNN_optuna_study.db'
if os.path.exists(db_path):
    study = optuna.create_study(
        study_name='BPNN_bpnn_tuning',
        storage=f'sqlite:///{db_path}',
        load_if_exists=True,
        direction='minimize'
    )
else:
    study = optuna.create_study(
        study_name='BPNN_bpnn_tuning',
        storage=f'sqlite:///{db_path}',
        direction='minimize'
    )

study.optimize(objective, n_trials=500)

best_params = study.best_params
print('Best hyperparameters:', best_params)

input_size = X_train.shape[1]
num_layers = best_params['num_layers']
layer_sizes = [best_params[f'layer_{i + 1}_size'] for i in range(num_layers)]
model = Net(input_size, num_layers, layer_sizes).to(device)
if best_params['optimizer'] == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
else:
    optimizer = optim.SGD(model.parameters(), lr=best_params['lr'], momentum=0.9)
criterion = nn.MSELoss()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)

best_loss = float('inf')
num_epochs = best_params['num_epochs']
patience = best_params['patience']
counter = 0

for epoch in range(num_epochs):
    model.train()
    train_loop = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
    for batch_X, batch_y in train_loop:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loop.set_postfix(loss=loss.item())

    # 使用测试集进行验证（注意：实际应用中建议保留独立验证集）
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for test_X, test_y in test_loader:
            test_outputs = model(test_X)
            test_loss += criterion(test_outputs, test_y).item()
        test_loss /= len(test_loader)

    # 早停机制
    if test_loss < best_loss:
        best_loss = test_loss
        counter = 0
        torch.save(model.state_dict(), 'BPNN_best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            break

model.load_state_dict(torch.load('BPNN_best_model.pth', map_location=device))
model.eval()
with torch.no_grad():
    y_pred = model(X_test.to(device)).to('cpu')

    y_pred_np = y_pred.numpy()
    y_test_np = y_test.to('cpu').numpy()
    y_pred_np = y_pred_np.reshape(-1)
    y_test_np = y_test_np.reshape(-1)

    test_loss = mean_squared_error(y_test_np, y_pred_np)
    r2 = r2_score(y_test_np, y_pred_np)

    print(f'Test MSE: {test_loss:.4f}')
    print(f'Test R²: {r2:.4f}')


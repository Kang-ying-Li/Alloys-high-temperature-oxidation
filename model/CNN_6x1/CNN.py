import matplotlib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(comment='_CNN')

data = pd.read_csv('../../data/data.csv')

input_features = ['Ti', 'Al', 'Sn', 'Zr', 'Mo', 'Si', 'C', 'Nb', 'Hf', 'Ta', 'Cr', 'B', 'Mn',
                  'V', 'Co', 'Fe', 'Ni', 'W', 'Cu', 'Zn', 'Mg', 'Gd', 'Y', 'P', 'Ce', 'N', 'Er',
                  'Ca', 'Ge', 'Re', 'O', 'Ga', 'T']
output_feature = ['lnkp']

X = data[input_features].values
y = data[output_feature].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=1145)

X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
y_train = np.nan_to_num(y_train, nan=0.0, posinf=1e6, neginf=-1e6)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
y_test = np.nan_to_num(y_test, nan=0.0, posinf=1e6, neginf=-1e6)

X_train = torch.from_numpy(X_train).float().to(device)
y_train = torch.from_numpy(y_train).float().squeeze().to(device)
X_test = torch.from_numpy(X_test).float().to(device)
y_test = torch.from_numpy(y_test).float().squeeze().to(device)

X_train = X_train.unsqueeze(1)
X_test = X_test.unsqueeze(1)

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


class CNN(nn.Module):
    def __init__(self, input_channels, kernel_sizes, filters, fc_units):
        super(CNN, self).__init__()

        self.conv_layers = nn.ModuleList()

        in_channels = input_channels
        for i in range(len(kernel_sizes)):
            self.conv_layers.append(
                nn.Conv1d(in_channels, filters[i],
                          kernel_size=kernel_sizes[i],
                          padding=kernel_sizes[i] // 2))
            self.conv_layers.append(nn.BatchNorm1d(filters[i]))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool1d(kernel_size=1))
            in_channels = filters[i]

        self.fc_layers = nn.ModuleList()
        prev_units = 1248
        for units in fc_units:
            self.fc_layers.append(nn.Linear(prev_units, units))
            self.fc_layers.append(nn.ReLU())
            prev_units = units
        self.fc_layers.append(nn.Linear(prev_units, 1))

    def forward(self, x):
        out = x
        for layer in self.conv_layers:
            out = layer(out)

        out = out.view(out.size(0), -1)

        for layer in self.fc_layers:
            out = layer(out)
        return out

kernel_sizes = [6, 6, 6, 6, 6, 6]
filters = [32, 32, 32, 32, 32, 32]
fc_units = [512, 256]

model = CNN(input_channels=1, kernel_sizes=kernel_sizes, filters=filters, fc_units=fc_units).to(device)

optimizer = optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.MSELoss()

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_epochs = 500
best_loss = float('inf')
scaler = GradScaler()
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_loop = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)

    for batch_X, batch_y in train_loop:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()

        with autocast():
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        train_loop.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)
    writer.add_scalar('Loss/Train', avg_train_loss, epoch)

    model.eval()
    val_loss = 0.0
    all_preds = []
    all_true = []
    with torch.no_grad():
        for val_X, val_y in test_loader:
            val_X, val_y = val_X.to(device), val_y.to(device)
            with autocast():
                outputs = model(val_X)
                val_loss += criterion(outputs, val_y.unsqueeze(1)).item()

            all_preds.append(outputs.cpu().numpy())
            all_true.append(val_y.cpu().numpy())

    val_loss /= len(test_loader)
    all_preds = np.concatenate(all_preds).flatten()
    all_true = np.concatenate(all_true).flatten()
    val_r2 = r2_score(all_true, all_preds)

    writer.add_scalar('Loss/Validation', val_loss, epoch)
    writer.add_scalar('R2/Validation', val_r2, epoch)
    print(
        f'Epoch [{epoch + 1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val R²: {val_r2:.4f}')

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), '1DCNN_best_model.pth')
        print(f"New best model saved with val loss: {val_loss:.4f}")

model.load_state_dict(torch.load('1DCNN_best_model.pth', map_location=device))
model.eval()
with torch.no_grad():
    y_pred = model(X_test.to(device)).to('cpu')

    y_pred_np = y_pred.numpy()
    y_test_np = y_test.to('cpu').numpy()
    y_pred_np = y_pred_np.reshape(-1)

    test_loss = mean_squared_error(y_test_np, y_pred_np)
    r2 = r2_score(y_test_np, y_pred_np)

    print(f'Test MSE: {test_loss:.4f}')
    print(f'Test R²: {r2:.4f}')

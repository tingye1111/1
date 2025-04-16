import xarray as xr
import pandas as pd
import pytz
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ========================
# 資料處理
# ========================
ds = xr.open_dataset("1-1-4.nc")
time_utc = pd.to_datetime(ds['valid_time'].values).tz_localize('UTC')
time_taiwan = time_utc.tz_convert('Asia/Taipei')
ds['valid_time'] = time_taiwan
target_lat, target_lon = 22.7304, 120.3125
ghi = ds["ssrd"].interp(latitude=target_lat, longitude=target_lon)
df = ghi.to_dataframe().reset_index()
df["GHI"] = df["ssrd"] / 1_000_000
df = df[["valid_time", "GHI"]].rename(columns={"valid_time": "Timestamp"})

if df["Timestamp"].dt.tz is None:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.tz_localize('Asia/Taipei')
else:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.tz_convert('Asia/Taipei')

print(df.head())

# ========================
# Dataset 類別
# ========================
class SolarDataset(Dataset):
    def __init__(self, df, time_steps=24, predict_steps=1, scaler=None):
        self.time_steps = time_steps
        self.predict_steps = predict_steps
        self.df = df
        self.scaler = scaler if scaler else MinMaxScaler()
        self.data = self.df["GHI"].values
        self.data = self.scaler.fit_transform(self.data.reshape(-1, 1)).reshape(-1)
        self.data = self.data.astype(np.float32)

    def __len__(self):
        return len(self.data) - self.time_steps - self.predict_steps

    def __getitem__(self, index):
        X = self.data[index:index + self.time_steps]
        y = self.data[index + self.time_steps : index + self.time_steps + self.predict_steps]
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(y, dtype=torch.float32)
        return X, y

# ========================
# LSTM 模型（無 Dropout）
# ========================
class LSTM(nn.Module):
    def __init__(self, input_feature_dim, hidden_feature_dim=128, hidden_layers=2, output_dim=24):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_feature_dim,
            hidden_size=hidden_feature_dim,
            num_layers=hidden_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_feature_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 128).to(x.device)
        c0 = torch.zeros(2, x.size(0), 128).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # 取最後時間步
        out = self.fc(out)
        return out

# ========================
# 訓練與驗證
# ========================
time_steps = 240
predict_steps = 10
batch_size = 32
num_epoch = 100

# 資料集與設備
dataset = SolarDataset(df, time_steps=time_steps, predict_steps=predict_steps)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 分割資料集
total_size = len(dataset)
train_size = int(total_size * 0.8)
val_size = int(total_size * 0.1)
test_size = total_size - train_size - val_size
train_dataset = Subset(dataset, range(0, train_size))
val_dataset = Subset(dataset, range(train_size, train_size + val_size))
test_dataset = Subset(dataset, range(train_size + val_size, total_size))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 模型、損失、優化器
model = LSTM(input_feature_dim=1, output_dim=predict_steps).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loss_history, val_loss_history = [], []

# ========================
# 訓練迴圈
# ========================
for epoch in range(num_epoch):
    model.train()
    train_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_loss_history.append(train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = criterion(output, y)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    val_loss_history.append(val_loss)

    print(f"Epoch [{epoch+1}/{num_epoch}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

# ========================
# 測試與評估
# ========================
preds, trues = [], []
model.eval()
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        output = model(X)
        preds.append(output.cpu().numpy().flatten())
        trues.append(y.cpu().numpy().flatten())

preds = np.concatenate(preds, axis=0)
trues = np.concatenate(trues, axis=0)
preds_original = dataset.scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
trues_original = dataset.scaler.inverse_transform(trues.reshape(-1, 1)).flatten()

rmse = np.sqrt(mean_squared_error(trues_original, preds_original))
mae = mean_absolute_error(trues_original, preds_original)
r2 = r2_score(trues_original, preds_original)

print(f"RMSE (MJ/m²): {rmse:.6f}")
print(f"MAE (MJ/m²): {mae:.6f}")
print(f"R²: {r2:.6f}")

# ========================
# 視覺化
# ========================
plt.figure(figsize=(10, 4))
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Validation Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("train_val_loss_direct.png")
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(trues_original[:720], label="True (MJ/m²)")
plt.plot(preds_original[:720], label="Predicted (MJ/m²)", linestyle="--")
plt.title("GHI 30-Day Direct Prediction (time Steps)")
plt.xlabel("Time Step")
plt.ylabel("GHI (MJ/m²)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ghi_30_day_direct_prediction.png")
plt.show()

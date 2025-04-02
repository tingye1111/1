import xarray as xr
import pandas as pd
import pytz
# 讀取資料
ds = xr.open_dataset("1-1-2.nc")

# 轉換時間為台灣時區
time_utc = pd.to_datetime(ds['valid_time'].values)
time_utc = time_utc.tz_localize('UTC')
time_taiwan = time_utc.tz_convert('Asia/Taipei')
ds['valid_time'] = time_taiwan

# 設定經緯度：使用插值
target_lat = 22.7304
target_lon = 120.3125

ghi = ds["ssrd"]
ghi_k = ghi.interp(latitude=target_lat, longitude=target_lon)

# 將 DataArray 轉換為 DataFrame
df = ghi_k.to_dataframe().reset_index()

# 單位轉換：J/m² 轉 MJ/m²
df["GHI"] = df["ssrd"] / 1_000_000

# 只保留時間與 GHI 欄位
df = df[["valid_time", "GHI"]]
df = df.rename(columns={"valid_time": "Timestamp"})
# 檢查結果
print(df)

# dataset 
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class SolarDataset(Dataset):
    def __init__(self, df, time_steps=24, scaler=None):
        self.time_steps=time_steps
        self.df=df
        self.scaler = scaler if scaler else MinMaxScaler()
        self.data=self.df["GHI"].values
        self.data=self.scaler.fit_transform(self.data.reshape(-1,1)).reshape(-1)
        self.data=self.data.astype(np.float32)
    def __len__(self):
        return len(self.data)-self.time_steps
    def __getitem__(self, index):
        X = self.data[index:index + self.time_steps]
        y = self.data[index + self.time_steps]
        X=torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        y=torch.tensor(y, dtype=torch.float32)
        return X, y

dataset=SolarDataset(df)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#LSTM model
from torch import nn
class LSTM(nn.Module):
    def __init__(self, input_feature_dim ,hidden_feature_dim=64,hidden_layers=2,output_dim=1):
        super(LSTM,self).__init__()
        self.input_feature_dim=input_feature_dim
        self.hidden_feature_dim=hidden_feature_dim
        self.hidden_layers=hidden_layers
        self.output_dim=output_dim
        #LSTM layer
        self.LSTM=nn.LSTM(input_feature_dim,hidden_feature_dim,hidden_layers,batch_first=True)
        #output layer
        self.fc=nn.Linear(hidden_feature_dim,output_dim)
    
    def forward(self,x):
        h0 = torch.zeros(self.hidden_layers, x.size(0), self.hidden_feature_dim).to(x.device)
        c0 = torch.zeros(self.hidden_layers, x.size(0), self.hidden_feature_dim).to(x.device)
        out, _ = self.LSTM(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    
#data loader 
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch.optim as optim
split_ratio=0.8
total_size=len(dataset)
train_size=int(total_size*split_ratio)
test_size=total_size-train_size

train_dataset=Subset(dataset,range(train_size))
test_dataset=Subset(dataset,range(train_size,total_size))
train_dataloader=DataLoader(train_dataset,batch_size=32,shuffle=False)
test_dataloader=DataLoader(test_dataset,batch_size=32,shuffle=False)


#Training loop
epoch=100
model = LSTM(input_feature_dim=1)
model = model.to(device)
criterion=nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()
loss_history=[]
for epoch in range(epoch):
    total_loss=0
    for X,y in train_dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X).squeeze()
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss=total_loss / len(train_dataloader)
    loss_history.append(avg_loss)
    print(f"Epoch [{epoch+1}/{epoch}] - Loss: {avg_loss:.6f}")

#Testing loop
model.eval()
test_loss=0
with torch.no_grad():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        output = model(X).squeeze()
        loss = criterion(output, y)
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_dataloader)
print(f"Test Loss: {avg_test_loss:.6f}")






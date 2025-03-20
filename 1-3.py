import xarray as xr
import pandas as pd
import pytz
ds = xr.open_dataset("1-1-1.nc")
time_utc = pd.to_datetime(ds['valid_time'].values)
# 假設原始時間是 UTC，將其設置為 UTC 時區
time_utc = time_utc.tz_localize('UTC')
# 轉換到台灣時區（UTC+8）
time_taiwan = time_utc.tz_convert('Asia/Taipei')

ds['valid_time']=time_taiwan
#設定經緯度
ghi = ds["ssrd"]
ghi_k = ghi.sel(latitude=23, longitude=120, method='nearest')
#將 DataArray 轉換為 DataFrame
df = ghi_k.to_dataframe().reset_index()
# 單位轉換：將 J/m² 轉為 W/m²
df["GHI"] = df["ssrd"]/1000000
# 只保留需要的欄位
df = df[["valid_time", "GHI"]]
# 重新命名欄位
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
import torch.optim as optim
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)




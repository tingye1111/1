import xarray as xr
import pandas as pd
import pytz
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
from vmdpy import VMD
warnings.filterwarnings('ignore')
def process_ghi_dataset(filepath, target_lat, target_lon):
    """Process NetCDF dataset for GHI data"""
    try:
        ds = xr.open_dataset(filepath)
        ds['valid_time'] = pd.to_datetime(ds['valid_time'].values).tz_localize('UTC').tz_convert('Asia/Taipei')
        ghi = ds["ssrd"].interp(latitude=target_lat, longitude=target_lon)
        df = ghi.to_dataframe().reset_index()
        df["GHI"] = (df["ssrd"] / 1_000_000).round(0)  # Convert to MJ/m²
        df = df[["valid_time", "GHI"]].rename(columns={"valid_time": "Timestamp"})
        df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.tz_convert("Asia/Taipei")
        df.set_index("Timestamp", inplace=True)

        # Calculate daily and monthly accumulation
        daily_ghi = df.resample("D").sum().reset_index().iloc[:-1]
        monthly_ghi = df.resample("ME").sum().reset_index().iloc[:-1]

        # === Daily outlier processing ===
        daily_ghi = daily_ghi[(daily_ghi["GHI"] > 0) & (daily_ghi["GHI"] < 45)]
        Q1 = daily_ghi["GHI"].quantile(0.25)
        Q3 = daily_ghi["GHI"].quantile(0.75)
        IQR = Q3 - Q1
        mask = (daily_ghi["GHI"] >= Q1 - 1.5 * IQR) & (daily_ghi["GHI"] <= Q3 + 1.5 * IQR)
        daily_ghi = daily_ghi[mask].reset_index(drop=True)
        daily_ghi["GHI"] = daily_ghi["GHI"].interpolate(method='linear')

        # === Monthly outlier processing ===
        monthly_ghi = monthly_ghi[(monthly_ghi["GHI"] >= 0) & (monthly_ghi["GHI"] <= 1200)]
        Q1_m = monthly_ghi["GHI"].quantile(0.25)
        Q3_m = monthly_ghi["GHI"].quantile(0.75)
        IQR_m = Q3_m - Q1_m
        mask_m = (monthly_ghi["GHI"] >= Q1_m - 1.5 * IQR_m) & (monthly_ghi["GHI"] <= Q3_m + 1.5 * IQR_m)
        monthly_ghi = monthly_ghi[mask_m].reset_index(drop=True)
        monthly_ghi["GHI"] = monthly_ghi["GHI"].fillna(method='bfill')

        return daily_ghi, monthly_ghi
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return pd.DataFrame(), pd.DataFrame()

def process_ghi_dataset_grib(filepath, target_lat, target_lon):
    """Process GRIB dataset for GHI data"""
    try:
        ds = xr.open_dataset(filepath, engine="cfgrib")

        # Find time column
        if 'time' in ds:
            time_col = 'time'
        elif 'valid_time' in ds:
            time_col = 'valid_time'
        else:
            raise ValueError("GRIB file does not contain suitable time column!")

        ds[time_col] = pd.to_datetime(ds[time_col].values).tz_localize('UTC').tz_convert('Asia/Taipei')

        # Find radiation variable
        ghi_var = 'ssrd' if 'ssrd' in ds else list(ds.data_vars)[0]
        ghi = ds[ghi_var].interp(latitude=target_lat, longitude=target_lon)

        df = ghi.to_dataframe().reset_index()
        df["GHI"] = df[ghi_var] / 1_000_000
        df = df[[time_col, "GHI"]].rename(columns={time_col: "Timestamp"})
        df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.tz_convert("Asia/Taipei")
        df.set_index("Timestamp", inplace=True)

        daily_ghi = df.resample("D").sum().reset_index().iloc[:-1]
        monthly_ghi = df.resample("ME").sum().reset_index().iloc[:-1]
        return daily_ghi, monthly_ghi
    except Exception as e:
        print(f"Error processing GRIB {filepath}: {e}")
        return pd.DataFrame(), pd.DataFrame()

def load_all_datasets(target_lat, target_lon):
    """Load and combine all datasets"""
    file_list = [
        ("1-1-4.nc", process_ghi_dataset),
        ("20232022GHI.nc", process_ghi_dataset),
        ("20212020GHI.nc", process_ghi_dataset),
        ("20192018GHI.nc", process_ghi_dataset),
        ("20172016GHI.nc", process_ghi_dataset),
        ("20152014GHI.nc", process_ghi_dataset),
        ("20132012GHI.nc", process_ghi_dataset),
        ("20112010GHI.nc", process_ghi_dataset),
        ("20092008GHI.nc", process_ghi_dataset),
        ("20072006GHI.nc", process_ghi_dataset),
        ("GRIB.grib", process_ghi_dataset_grib)
    ]
    
    daily_dfs = []
    monthly_dfs = []
    
    for filename, process_func in file_list:
        try:
            daily, monthly = process_func(filename, target_lat, target_lon)
            if not daily.empty and not monthly.empty:
                daily_dfs.append(daily)
                monthly_dfs.append(monthly)
                print(f"Successfully loaded {filename}")
            else:
                print(f"Warning: {filename} returned empty dataframes")
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
    
    if daily_dfs and monthly_dfs:
        combined_daily = pd.concat(daily_dfs, ignore_index=True).sort_values(by="Timestamp").reset_index(drop=True)
        combined_monthly = pd.concat(monthly_dfs, ignore_index=True).sort_values(by="Timestamp").reset_index(drop=True)
        
        # Remove duplicates
        combined_daily = combined_daily.drop_duplicates(subset=['Timestamp']).reset_index(drop=True)
        combined_monthly = combined_monthly.drop_duplicates(subset=['Timestamp']).reset_index(drop=True)
        
        return combined_daily, combined_monthly
    else:
        raise ValueError("No valid datasets were loaded!")


### Dataset ###
class SolarDatasetIMFSum(Dataset):
    def __init__(self, df, time_steps=24, predict_steps=1, scaler=None, K=6):
        self.time_steps = time_steps
        self.predict_steps = predict_steps
        self.df = df.copy()  # df 要包含 IMF1, ..., IMF_K, GHI_trend
        self.K = K

        if scaler is None:
            self.scaler = MinMaxScaler()
            self.data = self.scaler.fit_transform(self.df.values)
        else:
            self.scaler = scaler
            self.data = self.scaler.transform(self.df.values)
        self.data = self.data.astype(np.float32)
        self.n_features = self.data.shape[1]
        
    def __len__(self):
        return max(0, len(self.data) - self.time_steps - self.predict_steps + 1)

    def __getitem__(self, idx):
        # X shape: (time_steps, n_features)
        X = self.data[idx:idx+self.time_steps, :]
        # y shape: (predict_steps, K)  (只要 IMF，不要 GHI_trend)
        y = self.data[idx+self.time_steps:idx+self.time_steps+self.predict_steps, :self.K]
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return X, y

### Model ###
class BiLSTM(nn.Module):
    def __init__(self, input_feature_dim, hidden_feature_dim=90, hidden_layers=2, output_dim=6, predict_steps=1, dropout_rate=0.3):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_feature_dim,
            hidden_size=hidden_feature_dim,
            num_layers=hidden_layers,
            dropout=dropout_rate if hidden_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_feature_dim * 2, output_dim * predict_steps)
        self.predict_steps = predict_steps
        self.output_dim = output_dim

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # (batch, hidden*2)
        out = self.dropout(out)
        out = self.fc(out)
        out = out.view(-1, self.predict_steps, self.output_dim)  # (batch, predict_steps, K)
        return out

### Early Stopping ###
class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, verbose=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

### 訓練 ###
def train_model(model, train_loader, val_loader, num_epochs=200, patience=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=patience)
    train_loss_history, val_loss_history = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_count = 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)  # (batch, predict_steps, K)
            loss = criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            train_count += 1
        train_loss /= train_count
        train_loss_history.append(train_loss)
        
        model.eval()
        val_loss, val_count = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                loss = criterion(output, y)
                val_loss += loss.item()
                val_count += 1
        val_loss /= val_count
        val_loss_history.append(val_loss)
        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
    return train_loss_history, val_loss_history

### 多步評估 ###
def evaluate_model_sum_imfs(model, test_loader, scaler, device, K, predict_steps):
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            output = model(X)  # (batch=1, predict_steps, K)
            preds.append(output.cpu().numpy())
            trues.append(y.cpu().numpy())
    preds = np.concatenate(preds, axis=0)   # [N, predict_steps, K]
    trues = np.concatenate(trues, axis=0)   # [N, predict_steps, K]

    # inverse transform
    def inverse_transform_multi_column(scaler, data, col_indices):
        # data: [N, predict_steps, K]
        N, T, K = data.shape
        full = np.zeros((N * T, scaler.n_features_in_))
        for i, idx in enumerate(col_indices):
            full[:, idx] = data[:, :, i].reshape(-1)
        recovered = scaler.inverse_transform(full)
        return recovered[:, col_indices].reshape(N, T, K)
    col_indices = list(range(K))
    preds_original = inverse_transform_multi_column(scaler, preds, col_indices)
    trues_original = inverse_transform_multi_column(scaler, trues, col_indices)

    # 加總重構GHI
    preds_sum = np.sum(preds_original, axis=2)  # [N, predict_steps]
    trues_sum = np.sum(trues_original, axis=2)  # [N, predict_steps]

    # 計算每個步長的指標
    rmse_steps = [np.sqrt(mean_squared_error(trues_sum[:, i], preds_sum[:, i])) for i in range(predict_steps)]
    mae_steps = [mean_absolute_error(trues_sum[:, i], preds_sum[:, i]) for i in range(predict_steps)]
    r2_steps = [r2_score(trues_sum[:, i], preds_sum[:, i]) for i in range(predict_steps)]

    return preds_sum, trues_sum, rmse_steps, mae_steps, r2_steps

### 你原本的資料處理 function 這裡就省略，直接引用 ###

# 只要有 process_ghi_dataset, process_ghi_dataset_grib, load_all_datasets 等 function 即可

def main():
    target_lat, target_lon = 22.7304, 120.3125
    time_steps = 240
    predict_steps = 12   # 多步預測（例如預測未來12個月）
    batch_size = 32
    num_epochs = 200
    K = 6  # IMF數

    print("Loading datasets...")
    combined_daily, combined_monthly = load_all_datasets(target_lat, target_lon)

    # --- VMD ---
    ghi_signal = combined_monthly["GHI"].values.astype(float)
    imfs, _, _ = VMD(ghi_signal, alpha=2000, tau=0, K=K, DC=0, init=1, tol=1e-6)
    imf_df = pd.DataFrame(imfs.T, index=combined_monthly['Timestamp'], columns=[f'IMF{i+1}' for i in range(K)])
    imf_df['GHI_trend'] = ghi_signal

    # --- Dataset ---
    dataset = SolarDatasetIMFSum(imf_df, time_steps=time_steps, predict_steps=predict_steps, K=K)
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

    model = BiLSTM(input_feature_dim=imf_df.shape[1], output_dim=K, predict_steps=predict_steps)
    print("Starting training...")
    train_loss_history, val_loss_history = train_model(
        model, train_loader, val_loader, num_epochs=num_epochs, patience=25
    )

    # 評估 IMF 重構
    print("Evaluating model (IMF sum)...")
    preds_sum, trues_sum, rmse_steps, mae_steps, r2_steps = evaluate_model_sum_imfs(
        model, test_loader, dataset.scaler, torch.device("cuda" if torch.cuda.is_available() else "cpu"), K, predict_steps
    )

    # 印出每個步長的評估指標
    for i in range(predict_steps):
        print(f"Step {i+1}: RMSE={rmse_steps[i]:.4f}, MAE={mae_steps[i]:.4f}, R2={r2_steps[i]:.4f}")

    # 繪圖：顯示第1步和最後1步的預測結果
    plt.figure(figsize=(14, 6))
    plt.subplot(2, 1, 1)
    plt.plot(trues_sum[:, 0], label="True GHI (Step 1)")
    plt.plot(preds_sum[:, 0], "--", label="Predicted GHI (Step 1)")
    plt.legend()
    plt.title("Step 1")

    plt.subplot(2, 1, 2)
    plt.plot(trues_sum[:, -1], label=f"True GHI (Step {predict_steps})")
    plt.plot(preds_sum[:, -1], "--", label=f"Predicted GHI (Step {predict_steps})")
    plt.legend()
    plt.title(f"Step {predict_steps}")

    plt.tight_layout()
    plt.savefig('VMDTEST2_multistep.png')
    plt.show()

if __name__ == "__main__":
    main()


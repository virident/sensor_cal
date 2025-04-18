

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class SpatioTemporalRegressor(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=64, dropout_rate=0.3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )


    def forward(self, x):
        return self.model(x).squeeze(-1)  # flatten output to [batch_size]


def spatial_sincos_encoding(lat, lon, dim=16):
    """
    Encodes latitude and longitude into sinusoidal positional encodings.

    Args:
        lat (float): Latitude value in degrees.
        lon (float): Longitude value in degrees.
        d_model (int): Dimension of the positional encoding.

    Returns:
        numpy.ndarray: Sinusoidal positional encoding of shape (2, d_model).
    """
    position = np.array([lat, lon])  # Latitude and longitude as a 2D array
    
    encoding = np.zeros((2, dim))
    
    for i in range(0, dim, 2):
        encoding[0, i] = np.sin(position[0] / (10000 ** (i / dim)))
        encoding[0, i + 1] = np.cos(position[0] / (10000 ** (i / dim)))
        encoding[1, i] = np.sin(position[1] / (10000 ** (i / dim)))
        encoding[1, i + 1] = np.cos(position[1] / (10000 ** (i / dim)))
        
    return encoding

def prepare_temporal_data_with_positional_embedding(longitudes, latitudes, D300_vals, Ref_CO2_vals, embed_dim=16, seed=42):
    """
    Predict D300[t+1] - Ref_CO2[t+1] using D300[t], Ref_CO2[t], and spatial positional encoding of lat/lon at t.
    """
    n_samples = len(D300_vals) - 1  # because we are predicting t+1 from t

    # Generate spatial embeddings for each position
    # pos_embeds = np.array([
    #     spatial_sincos_encoding(latitudes[i], longitudes[i], dim=embed_dim)
    #     for i in range(n_samples)
    # ])

    # Create inputs: positional embedding + D300[t] + Ref_CO2[t]
    sensor_features = np.stack([
        D300_vals[:-1],
        Ref_CO2_vals[:-1]
    ], axis=1)

    X = np.hstack([longitudes, latitudes, sensor_features])

    # Target: calibration delta at t+1
    D300_arr = np.array(D300_vals)
    Ref_arr = np.array(Ref_CO2_vals)
    y = D300_arr[1:] - Ref_arr[1:]

    # Train/Val/Test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

    return X_train, y_train, X_val, y_val, X_test, y_test





def normalize_inputs(X):
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    X_norm = (X - mins) / (maxs - mins + 1e-8)
    return X_norm, mins, maxs

def train_temporal_model(X, y, epochs=1000, lr=1e-3):
    # Build X[t] -> D300[t+1] pairs
    X_norm, mins, maxs = normalize_inputs(X)

    X = torch.tensor(X_norm, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)

    model = SpatioTemporalRegressor(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        preds = model(X)
        loss = loss_fn(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model, mins, maxs

def predict(self, lon, lat, mins, maxs):
    inp = np.array([lon, lat])
    norm = (inp - mins) / (maxs - mins + 1e-8)
    x_tensor = torch.tensor(norm[None, :], dtype=torch.float32)
    
    with torch.no_grad():
        prediction = self.model(x_tensor)
    
    return prediction.item()

def parse_uniform_rows_from_csvs(folder_path, name1='D300', name2='Ref.CO2'):
    """
    Reads all CSV files in a folder and returns uniform row values (non-NaN in name2).
    
    Args:
        folder_path (str): Path to the folder containing CSV files.
        name1 (str): First column to extract (e.g., 'D300').
        name2 (str): Column used to filter non-NaN rows (e.g., 'Ref_CO2').

    Returns:
        D300_all_cols_uniform_rows: list of numpy arrays
        Ref_CO2_all_cols_uniform_rows: list of numpy arrays
    """
    # Step 1: Gather all file paths
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    all_valid_indices = None

    # Step 2: Find common valid row indices (non-NaN in name2)
    for filename in csv_files:
        df = pd.read_csv(os.path.join(folder_path, filename))
        if name2 not in df.columns:
            raise ValueError(f"Column '{name2}' not found in file {filename}")
        valid_indices = df[~df[name2].isna()].index

        if all_valid_indices is None:
            all_valid_indices = set(valid_indices)
        else:
            all_valid_indices &= set(valid_indices)  # intersection

    common_indices = sorted(list(all_valid_indices))

    # Step 3: Extract values from both columns using common indices
    D300_all_cols_uniform_rows = []
    Ref_CO2_all_cols_uniform_rows = []
    Long_all_cols_uniform_rows = []
    Lat_all_cols_uniform_rows = []


    for filename in csv_files:
        df = pd.read_csv(os.path.join(folder_path, filename))
        D300_all_cols_uniform_rows.append(df.loc[common_indices, name1].to_numpy())
        Ref_CO2_all_cols_uniform_rows.append(df.loc[common_indices, name2].to_numpy())
        Long_all_cols_uniform_rows.append(df.loc[common_indices, 'longitude'].to_numpy())
        Lat_all_cols_uniform_rows.append(df.loc[common_indices, 'latitude'].to_numpy())
    print("longs first element is ")
    print(Long_all_cols_uniform_rows[0])
    print("longs second element is ")
    print(Long_all_cols_uniform_rows[1])


    return D300_all_cols_uniform_rows, Ref_CO2_all_cols_uniform_rows, Long_all_cols_uniform_rows, Lat_all_cols_uniform_rows




def visualize_temporal_predictions(model, X_raw, y_true, mins, maxs, title="Next D300 Prediction"):
    X_norm = (X_raw - mins) / (maxs - mins + 1e-8)
    X_tensor = torch.tensor(X_norm, dtype=torch.float32)

    with torch.no_grad():
        preds = model(X_tensor).numpy()

    plt.figure(figsize=(8, 6))
    plt.plot(y_true, label="Actual D300[t+1]", marker='o')
    plt.plot(preds, label="Predicted", marker='x')
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("D300")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    d300, ref_co2, long, lat = parse_uniform_rows_from_csvs('sample_data_small')

    model = SpatioTemporalRegressor(input_dim=18, hidden_dim=64, dropout_rate=0.3)

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_temporal_data_with_positional_embedding(
        long, lat, d300, ref_co2, embed_dim=16
    )

    X_train_norm, mins, maxs = normalize_inputs(X_train[:, -2:])

    X_train_final = np.hstack([X_train[:, :-2], X_train_norm])
    X_val_final = np.hstack([X_val[:, :-2], (X_val[:, -2:] - mins) / (maxs - mins + 1e-8)])
    X_test_final = np.hstack([X_test[:, :-2], (X_test[:, -2:] - mins) / (maxs - mins + 1e-8)])


    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    model, mins, maxs = model.train_model(X_train_tensor, y_train_tensor)

    model.visualize_temporal_predictions(
        model, X_val_final, y_val, mins, maxs, title="Validation Set Predictions"
    )

    


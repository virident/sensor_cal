import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
from sklearn.metrics import mean_squared_error as MSE





def create_model(input_dim, hidden_dim, output_dim, dropout_rate, l1_lambda=0.1):
    model = xgb.XGBRegressor(objective='reg:squarederror',
                            n_estimators=1000,
                            learning_rate=0.01,
                            max_depth=5,
                            min_child_weight=1,
                            gamma=0,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            reg_alpha=0.005,
                            random_state=42,
                            n_jobs=-1)
    return model

def spatial_sincos_encoding(lat, lon, dim=16):
    """
    Encodes latitude and longitude into sinusoidal positional encodings.

    Args
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

def prepare_temporal_data_with_positional_embedding(longitudes, latitudes, D300_vals, Ref_CO2_vals, int_temp, ext_temp, int_humid, ext_humid, num_files, embed_dim=16, seed=42):
    """
    Prepare data where X has 3 columns (D300, longitude, latitude) for each file,
    and y has the corresponding D300 - Ref_CO2 values
    """
    # First, let's concatenate the data from all files
    X_list = []
    y_list = []
    
    # Loop through each file's data (they're in corresponding positions in the lists)
    print("num files is ", num_files)
    for i in range(0, num_files):
        # Create X matrix for this file: [D300, longitude, latitude]
        #version 1 with one offset
        # X_list.append(D300_vals[i][:-1])
        # X_list.append(longitudes[i][:-1])
        # X_list.append(latitudes[i][:-1])
        # X_list.append(int_temp[i][:-1])
        # X_list.append(ext_temp[i][:-1])
        # X_list.append(int_humid[i][:-1])
        # X_list.append(ext_humid[i][:-1])
    
        # # Create y values for this file
        # y_list.append(D300_vals[i][1:] - Ref_CO2_vals[i][1:])
        
        X_list.append(D300_vals[i])
        X_list.append(longitudes[i])
        X_list.append(latitudes[i])
        X_list.append(int_temp[i])
        X_list.append(ext_temp[i])
        X_list.append(int_humid[i])
        X_list.append(ext_humid[i])
    
        # Create y values for this file
        y_list.append(Ref_CO2_vals[i] - D300_vals[i])
    
    # Concatenate all files' data

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"X shape: {X.shape}")
    X = X.T
    y= y.T

    
    
    print(f"Final X shape: {X.shape}")  # Should be (total_samples, 3)
    print(f"Final y shape: {y.shape}")  # Should be (total_samples,)
    
    return X, y

def normalize_column(data, min_val, max_val):
    """
    Normalize a 1D array (column) to range [0, 1] given min and max.

    Parameters:
        data (np.ndarray): 1D array of values to normalize
        min_val (float): Minimum possible value (x)
        max_val (float): Maximum possible value (y)

    Returns:
        np.ndarray: Normalized data in range [0, 1]
    """
    return (data - min_val) / (max_val - min_val)

def scale_columns_to_range(data, col_maxs, col_mins, min_val=-0.5, max_val=0.5):
    """
    Scale each column of the input data to the specified range [min_val, max_val].
    
    Args:
        data (np.ndarray): Input data array
        min_val (float): Minimum value of the target range
        max_val (float): Maximum value of the target range
        
    Returns:
        np.ndarray: Scaled data
        tuple: (min_values, max_values) for each column
    """
    # Calculate min and max for each column
    # col_mins = np.min(data, axis=0)
    # col_maxs = np.max(data, axis=0)
    
    # Scale each column to [-0.5, 0.5]
    scaled_data = np.zeros_like(data)
    print("data shape is", data.shape)
    
    for i in range(len(col_mins)):
        col_range = col_maxs[i] - col_mins[i]
        if col_range == 0:  # Handle case where all values are the same
            scaled_data[:, i] = 0
        else:
            scaled_data[:, i] = min_val + (data[:, i] - col_mins[i]) * (max_val - min_val) / col_range
    print("scaled data shape is ", scaled_data.shape)
    return scaled_data, (col_mins, col_maxs)

def train_temporal_model(X_train, y_train, X_val, y_val, epochs=100, lr=1e-3, col_maxs=None, col_mins=None):
    # Scale input data
    X_scaled, (col_mins, col_maxs) = scale_columns_to_range(X_train, col_maxs, col_mins)
    
    # Convert data to tensors once at the beginning
   
    
    print("shape of X is ", X_scaled.shape)
    print("shape of y is ", y_train.shape)
    losses = []
    callback = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='models/weights.{epoch:02d}-{val_loss:.2f}.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    xgb_train = xgb.DMatrix(data = X_train, label = y_train, enable_categorical=True)
    xgb_val = xgb.DMatrix(data = X_val, label = y_val, enable_categorical=True)
    
    param = {"booster": "gblinear", "objective": "reg:linear"}
    xgb_r = xgb.train(params = param, dtrain = xgb_train, evals = [(xgb_train, "X_train"), (xgb_val, "validation")], num_boost_round = 200, early_stopping_rounds = 50) 
    pred = xgb_r.predict(xgb_val, iteration_range = (0, xgb_r.best_iteration)) 

    final_loss = np.sqrt(MSE(y_val, pred)) 
    print("RMSE : % f" %(final_loss)) 

    # model = create_model(input_dim=X_scaled.shape[1], hidden_dim=(X_scaled.shape[1])// 2, output_dim=y_train.shape[1], dropout_rate=0.4, l1_lambda=0.1)
    # #for x in range(0, X_scaled.shape[0]):
    # model.fit(X_scaled, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    # #version with early stopping
    # #model.fit(X_scaled, y_train, epochs=100, batch_size=32, callbacks=[callback], validation_data=(X_val, y_val), verbose=1)

    # model.predict(X_val[x], y_val[x], verbose=1)
   
#     plt.figure(figsize=(10, 6))
#     plt.plot(history.history['loss'], label='Training Loss', marker='o')
#     plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
#    # plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
#     plt.title('Loss Over Epochs')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

   
    # print("Final loss: ", final_loss)
    #print("Train loss: ", train_loss)
    losses = final_loss
    
    return xgb_r, losses

# def predict(self, lon, lat, mins, maxs):
#     inp = np.array([lon, lat])
#     norm = (inp - mins) / (maxs - mins + 1e-8)
#     x_tensor = torch.tensor(norm[None, :], dtype=torch.float32)
    
#     with torch.no_grad():
#         prediction = self.model(x_tensor)
    
#     return prediction.item()

def parse_uniform_rows_from_csvs(folder_path, name1='D300', name2='Ref.CO2'):
    """
    Reads all CSV files in a folder and returns uniform row values (non-NaN in both name1 and name2).
    
    Args:
        folder_path (str): Path to the folder containing CSV files.
        name1 (str): First column to extract (e.g., 'D300').
        name2 (str): Second column to extract (e.g., 'Ref.CO2').

    Returns:
        D300_all_cols_uniform_rows: list of numpy arrays
        Ref_CO2_all_cols_uniform_rows: list of numpy arrays
        Long_all_cols_uniform_rows: list of numpy arrays
        Lat_all_cols_uniform_rows: list of numpy arrays
    """
    # Step 1: Gather all file paths
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    all_valid_indices = None
    num_files = 0
    # Step 2: Find common valid row indices (non-NaN in both name1 and name2)
    # for filename in csv_files:
    #     num_files +=1
    #     df = pd.read_csv(os.path.join(folder_path, filename))
        
    #     # Check if required columns exist
    #     if name1 not in df.columns or name2 not in df.columns:
    #         raise ValueError(f"Columns '{name1}' or '{name2}' not found in file {filename}")
        
    #     df = df.ffill()
    #     # Find rows where both D300 and Ref.CO2 are valid (not NaN and not invalid)
    #     valid_indices = df[
    #         (~df[name1].isna()) &  # D300 is not NaN
    #         (~df[name2].isna()) &  # Ref.CO2 is not NaN
    #         (~df['latitude'].isna()) &  # latitude is not NaN
    #         (~df['longitude'].isna()) &  # longitude is not NaN
    #         (~df['SHT31HE'].isna()) & 
    #         (~df['SHT31TE'].isna()) & 
    #         (~df['SHT31HI'].isna()) & 
    #         (~df['SHT31TI'].isna()) & 
    #         (df[name1] != 0) &     # D300 is not 0 (assuming 0 is invalid)
    #         (df[name1] > -999) &   # D300 is not unreasonably negative
    #         (df[name1] < 10000)    # D300 is not unreasonably large
    #     ].index

    #     if all_valid_indices is None:
    #         all_valid_indices = set(valid_indices)
    #     else:
    #         all_valid_indices &= set(valid_indices)  # intersection

    # common_indices = sorted(list(all_valid_indices))
    

    #print("all valid indices shape is ", all_valid_indices.shape())

    # Step 3: Extract values from all columns using common indices
    D300_all_cols_uniform_rows = []
    Ref_CO2_all_cols_uniform_rows = []
    Long_all_cols_uniform_rows = []
    Lat_all_cols_uniform_rows = []
    Internal_Temp_all_cols_uniform_rows = []
    External_Temp_all_cols_uniform_rows = []
    Internal_Humidity_all_cols_uniform_rows = []
    External_Humidity_all_cols_uniform_rows = []

    max_entries = 100000000
    for filename in csv_files:
        num_files +=1
        df = pd.read_csv(os.path.join(folder_path, filename))
        df = df.ffill()
        maxes = []
        maxes.append(len((df.loc[:, [name1]].to_numpy())))
        maxes.append(len((df.loc[:, [name2]].to_numpy())))
        maxes.append(len((df.loc[:, ['longitude']].to_numpy())))
        maxes.append(len((df.loc[:, ['latitude']].to_numpy())))
        maxes.append(len((df.loc[:, ['SHT31TI']].to_numpy())))
        maxes.append(len((df.loc[:, ['SHT31TE']].to_numpy())))
        maxes.append(len((df.loc[:, ['SHT31HI']].to_numpy())))
        maxes.append(len((df.loc[:, ['SHT31HE']].to_numpy())))

        if max(maxes) < max_entries:
            max_entries = max(maxes)
        
    print("max entries is ", max_entries)
        

    for filename in csv_files:
        df = pd.read_csv(os.path.join(folder_path, filename))
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        D300_all_cols_uniform_rows.append((df[name1].head(max_entries)).to_numpy())
        Ref_CO2_all_cols_uniform_rows.append((df[name2].head(max_entries)).to_numpy())
        Long_all_cols_uniform_rows.append((df['longitude'].head(max_entries)).to_numpy())
        Lat_all_cols_uniform_rows.append((df['latitude'].head(max_entries)).to_numpy())
        Internal_Temp_all_cols_uniform_rows.append((df['SHT31TI'].head(max_entries)).to_numpy())
        External_Temp_all_cols_uniform_rows.append((df['SHT31TE'].head(max_entries)).to_numpy())
        Internal_Humidity_all_cols_uniform_rows.append((df['SHT31HI'].head(max_entries)).to_numpy())
        External_Humidity_all_cols_uniform_rows.append((df['SHT31HE'].head(max_entries)).to_numpy())
    
    maxes = []
    mins = []
    maxes.append(np.max(D300_all_cols_uniform_rows))
    mins.append(np.min(D300_all_cols_uniform_rows))
    maxes.append(np.max(Ref_CO2_all_cols_uniform_rows))
    mins.append(np.min(Ref_CO2_all_cols_uniform_rows))
    maxes.append(np.max(Long_all_cols_uniform_rows))
    mins.append(np.min(Long_all_cols_uniform_rows))
    maxes.append(np.max(Lat_all_cols_uniform_rows))
    mins.append(np.min(Lat_all_cols_uniform_rows))
    maxes.append(np.max(Internal_Temp_all_cols_uniform_rows))
    mins.append(np.min(Internal_Temp_all_cols_uniform_rows))
    maxes.append(np.max(External_Temp_all_cols_uniform_rows))
    mins.append(np.min(External_Temp_all_cols_uniform_rows))
    maxes.append(np.max(Internal_Humidity_all_cols_uniform_rows))
    mins.append(np.min(Internal_Humidity_all_cols_uniform_rows))
    maxes.append(np.max(External_Humidity_all_cols_uniform_rows))
    mins.append(np.min(External_Humidity_all_cols_uniform_rows))

    # Print some statistics about the data
    print(f"Shape of D300_all_cols: {np.array(D300_all_cols_uniform_rows).shape}")
    print(f"Number of files processed: {len(csv_files)}")
    #print(f"Number of valid rows per file: {len(common_indices)}")
    print(f"D300 value ranges:")
    for i, d300_vals in enumerate(D300_all_cols_uniform_rows):
        print(f"  File {i+1}: min={d300_vals.min():.2f}, max={d300_vals.max():.2f}, mean={d300_vals.mean():.2f}")

    return D300_all_cols_uniform_rows, Ref_CO2_all_cols_uniform_rows, Long_all_cols_uniform_rows, Lat_all_cols_uniform_rows, Internal_Temp_all_cols_uniform_rows, External_Temp_all_cols_uniform_rows, Internal_Humidity_all_cols_uniform_rows, External_Humidity_all_cols_uniform_rows, num_files, maxes, mins




def visualize_temporal_predictions(losses, title="Next D300 Prediction"):
    

    
    plt.plot(losses, label="Actual D300[t+1]", marker='o')
   
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("D300")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def save_model(model, filename = 'trained_model.pth', save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)
    save_path=os.path.join(save_dir, filename)
    model.save_model('xgboost_models/xgmodel.json')
    #model.export('models')
    
    print("saved model to {save_path}")

def split_train_val_test(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_ratio), random_state=random_state
    )
    
    # Calculate the proportion of validation and test relative to the temp set
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    
    # Split temp into validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_test_ratio), random_state=random_state
    )
    
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":

    d300, ref_co2, long, lat, int_temp, ext_temp, int_humid, ext_humid, num_files, maxes, mins = parse_uniform_rows_from_csvs('large_data_folder/sensing_data')

    #model = SpatioTemporalRegressor(input_dim=18, hidden_dim=64, dropout_rate=0.3)

    X_train, y_train = prepare_temporal_data_with_positional_embedding(
        long, lat, d300, ref_co2, int_temp, ext_temp, int_humid, ext_humid, num_files=num_files
    )

    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(X_train, y_train)
    
    print("Train shapes:", X_train.shape, y_train.shape)
    print("Validation shapes:", X_val.shape, y_val.shape)
    print("Test shapes:", X_test.shape, y_test.shape)
    
    print("shape of X_train in main is ")
    print(X_train.shape)
    print(y_train.shape)

    X_trans = X_train.T
    # for col in range(X_trans.shape[0]):
    #     max_value = max(X_train[col])
    #     min_value = min(X_train[col])
    #     X_train[col] = normalize_column(X_train[col], min_value, max_value)

    # X_train_final = np.hstack([X_train[:, :-2], X_train_norm])
    # X_val_final = np.hstack([X_val[:, :-2], (X_val[:, -2:] - mins) / (maxs - mins + 1e-8)])
    # X_test_final = np.hstack([X_test[:, :-2], (X_test[:, -2:] - mins) / (maxs - mins + 1e-8)])


    print(X_train.shape)
    print(y_train.shape)

    print("mins shape is ", len(mins))
    print("maxes shape is ", len(maxes))

    model, losses = train_temporal_model(X_train, y_train, X_val, y_val, epochs=50, lr=1e-10, col_maxs=maxes, col_mins=mins)
    
    
    save_model(model, filename="xgmodel.json")
    
    
    model_loaded = xgb.Booster()
    model_loaded.load_model('xgboost_models/xgmodel.json')
    
    
    

    # print("losses are ")
    # print(losses)


    test_dmatrix = xgb.DMatrix(data = X_test, label = y_test) 


    results = model.predict(test_dmatrix, iteration_range = (0, model.best_iteration))
    test_loss = np.sqrt(MSE(y_test, results))

    print("RMSE : % f" %(test_loss)) 

    print("sample at index 0")
    print("predicted value is ", results[0])
    print("actual value is ", y_test[0])
    


    # Visualize the training loss
    #plt.plot(losses, label="Training Loss", marker='o')


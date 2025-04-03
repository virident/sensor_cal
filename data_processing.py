import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import os
from sklearn import linear_model

# file_dict = os.fsencode("senseurcity_data_v02/dataset")

# ant_df_list = []

# for file in os.listdir(file_dict):
#     filename = os.fsdecode(file)
#     if filename.startswith("Antwerp") or filename.startswith('ANT'):
#         df = pd.read_csv(r'senseurcity_data_v02/dataset/' + filename)
#         df = df.replace("NaN", np.nan)

#         df_ffill = df.ffill()
#         df = df_ffill.bfill()
#         ant_df_list.append(df)

ref_df = pd.read_csv('senseurcity_data_v02/dataset/ANT_REF_R801_Fidas_UTC.csv')

df = pd.read_csv('senseurcity_data_v02/dataset/Antwerp_402B00.csv')
ref_df = df.iloc[:, 77:]
df = df.iloc[:, :77]

df = df.replace("NaN", np.nan)

df_ffill = df.ffill()
df = df_ffill.bfill()
df = df.replace('W', np.nan)
# df=df.replace('T.min', np.nan)
# df=df.replace('T.max', np.nan)
# df=df.replace('Rh.min', np.nan)
# df=df.replace('Rh.max', np.nan)
# df=df.replace('Low_values', np.nan)
# df=df.replace('High_values', np.nan)
# df=df.replace('OutliersMin', np.nan)
# df=df.replace('OutliersMax', np.nan)
# df = df.replace("Inv", np.nan)

df=df.dropna(axis=0)

print(df)


print(ref_df)



model = linear_model.LinearRegression()


# print(df_clean["Absolute_humidity"][0])


# print(type(df_clean["Absolute_humidity"][0]))


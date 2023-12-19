import numpy as np
import pandas as pd

path = '../data/ours/re15.csv'

df = pd.read_csv(path)
# print(len(df))

cols = list(df.columns)
cols.remove("Datetime")
cols.remove("FCS_S05_L2_018")

df = df[['Datetime'] + cols + ['FCS_S05_L2_018']]

cols_data = df.columns[1:]
df_data = df[cols_data]
print(df_data)
print("df data mean %f" % np.mean(df_data.values))
print("df data std %f" % np.std(df_data.values))
s = (df_data.values - np.mean(df_data.values)) / np.std(df_data.values)
print(s)
print(np.std(df_data.values))
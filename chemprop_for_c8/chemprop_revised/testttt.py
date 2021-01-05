import pandas as pd
import numpy as np

file = pd.read_csv('./saved_models/pred_output/seed60_test/grad_m.csv')
header = file.columns
file = file.values
print(file[:5, :])
print(file.shape)
file = file.flatten()
print(file[:10])
file = file.reshape((2, -1))
print(file[:, :5])
print(file.T)
pd.DataFrame(file.T, columns=header, index=None).\
    to_csv('./saved_models/pred_output/seed60_test/grad_m.csv', index=False)
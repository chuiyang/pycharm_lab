from sklearn.datasets import load_boston
import pandas as pd
boston = load_boston()
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names

x = boston.data
y = boston.target
print(x.shape)
print(len(data.columns))
print(y.reshape((1, -1)).shape)

pd.DataFrame(x, columns=data.columns, index=None).to_csv('./../../../pycharm_gene/matlab/boston_x.csv', index=False)
pd.DataFrame(y.reshape((-1, 1)), columns=['price'], index=None).to_csv('./../../../pycharm_gene/matlab/boston_y.csv', index=False)

print(boston)

#multivariable regression and polynomial regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import seaborn as sns

'''from sklearn.datasets import load_boston
data=load_boston()
print(data.keys())

boston = pd.DataFrame(data.data, columns = data.feature_names)
print(boston.head())
boston["MEDV"] = data.target'''

from sklearn.datasets import fetch_openml
boston = fetch_openml(name='boston',version=1,as_frame=True)

#X = boston[['LSTAT','RM']]
#y=boston["MEDV"]
X=boston.data[["LSTAT",'RM']]
y=boston.target
print(X.head())
print(y.head())

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=5)

from sklearn.linear_model import LinearRegression
linear_mdl=LinearRegression()
linear_mdl.fit(X_train,y_train)

from sklearn.metrics import mean_squared_error

y_test_predict = linear_mdl.predict(X_test)

rmse_linear_mdl=(np.sqrt(mean_squared_error(y_test,y_test_predict)))
print("THE rmse is case of multivariable regression is",rmse_linear_mdl)

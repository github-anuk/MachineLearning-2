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

iris=pd.read_csv("iris.csv")

iris['species']=iris['species'].map({'setosa':0,"versicolor":1,'virginica':2,})
print(iris.head())

#X = boston[['LSTAT','RM']]
#y=boston["MEDV"]
X=iris[["sepal_length","sepal_width","petal_length","petal_width"]]
y=iris['species']
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

#polynomial regression 

from sklearn.preprocessing import PolynomialFeatures
poly_features= PolynomialFeatures(degree = 2)

X_train_poly= poly_features.fit_transform(X_train)

poly_model = LinearRegression()
poly_model.fit(X_train_poly,y_train)

X_test_poly = poly_features.fit_transform(X_test)
y_test_predict_poly = poly_model.predict(X_test_poly)

rmse_poly_model=(np.sqrt(mean_squared_error(y_test,y_test_predict_poly)))
print("the rmse in case of polynomial regression is ", rmse_poly_model)
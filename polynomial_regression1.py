# -*- coding: utf-8 -*-
"""
Created on Wed May 23 15:47:47 2018

@author: achyu
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#reading csv files
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,-1].values
#missing data handling
"""from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])
#categorical data encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x=LabelEncoder()
X[:,0]=labelencoder_x.fit_transform(X[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)"""
#traing data and testing data
"""from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)"""
#feature scalin
"""from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)"""
#fitting linear regression into dataset
from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(X,y)
#fitting polynomial regression to data set
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(X)
linreg2=LinearRegression()
linreg2.fit(x_poly,y)
#visualizing linear regression result
plt.scatter(X,y,c='r')
plt.plot(X,linreg.predict(X),c='b')
plt.title('Truth or bluff(linear)')
plt.xlabel('levels')
plt.ylabel('Salary')
plt.show()
#visualizing polynomial regression result
x_grid=np.arange(min(X),max(X)+0.1,0.1)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(X,y,c='r')
plt.plot(x_grid,linreg2.predict(poly_reg.fit_transform(x_grid)),c='b')
plt.title('Truth or bluff(polynomial)')
plt.xlabel('levels')
plt.ylabel('Salary')
plt.show()

linreg.predict(6.5)
linreg2.predict(poly_reg.fit_transform(6.5))
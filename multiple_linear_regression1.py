# -*- coding: utf-8 -*-
"""
Created on Mon May 21 15:28:22 2018

@author: achyu
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#reading csv files
dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
#missing data handling
"""from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])"""
#categorical data encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x=LabelEncoder()
X[:,3]=labelencoder_x.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

#Avoiding dummy variable trap
X=X[:,1:]

#traing data and testing data
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#feature scalin"""
"""from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)"""

#Fitting multiple linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
#predicting test data
y_pred=regressor.predict(x_test)
#visualizing
import statsmodels.formula.api as sm
X=np.append(np.ones((50,1)).astype(int),X,1)
x_opt=X[:,:]
regressor_ols=sm.OLS(y,x_opt).fit()
regressor_ols.summary()
x_opt=X[:,[0,1,3,4,5]]
regressor_ols=sm.OLS(y,x_opt).fit()
regressor_ols.summary()
x_opt=X[:,[0,3,4,5]]
regressor_ols=sm.OLS(y,x_opt).fit()
regressor_ols.summary()
x_opt=X[:,[0,3,5]]
regressor_ols=sm.OLS(y,x_opt).fit()
regressor_ols.summary()
x_opt=X[:,[0,3]]
regressor_ols=sm.OLS(y,x_opt).fit()
regressor_ols.summary()


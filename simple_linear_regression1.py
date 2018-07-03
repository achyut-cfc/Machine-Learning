# -*- coding: utf-8 -*-
"""
Created on Mon May 21 13:21:34 2018

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
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values
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
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)
#feature scalin
"""from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)"""

#fitting simple linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
#predicting test result
y_pred=regressor.predict(x_test)
#visualizing the data set

plt.scatter(x_train,y_train,c='r')
plt.plot(x_train,regressor.predict(x_train),c='b')
plt.title('Salaryvs experience(training set)')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_test,y_test,c='b')
plt.plot(x_train,regressor.predict(x_train),c='b')
plt.title('Salaryvs experience(test set)')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()

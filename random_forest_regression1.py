# -*- coding: utf-8 -*-
"""
Created on Thu May 24 15:25:07 2018

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

#fitting regression to data set
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(X,y)
#predicting result
y_pred=regressor.predict(6.5)

#visualizing  regression result
x_grid=np.arange(min(X),max(X)+0.01,0.01)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(X,y,c='r')
plt.plot(x_grid,regressor.predict(x_grid),c='b')
plt.title('Truth or bluff(decision tree)')
plt.xlabel('levels')
plt.ylabel('Salary')
plt.show()


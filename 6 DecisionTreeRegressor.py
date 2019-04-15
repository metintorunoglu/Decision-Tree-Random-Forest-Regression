# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 10:59:47 2019

@author: nesib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import the dataset
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:, 1:2].values
y=dataset.iloc[:, 2].values

#Split (no need)
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)"""

#Feature Scaling (no need, will be implemented automaticly)
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)""


#fitting the Decision Tree Regression model to dataset
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

#prediction
y_predict=regressor.predict(X)

￼#visualising Decision Tree Regression results(High resulation)
X_grid=np.arange(min(X), max(X), 0.1)
X_grid=X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluf (DecisionTreeRegression Model)')
plt.xlabel=('position level')
plt.ylabel=('salary')
plt.show()
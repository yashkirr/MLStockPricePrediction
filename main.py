# Author: Yashkir Ramsamy
# Contact: me@yashkir.co.za
# Date: 2021/01/12

import quandl
import ApiKeys
import investpy
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

forecast = 30 #days
print("nice!")
quandl.ApiConfig.api_key = ApiKeys.quandl_key # sets the API key for quandl's service
frame = quandl.get("WIKI/AMZN")
frame = frame[['Adj. Close']]
frame['Prediction'] = frame[['Adj. Close']].shift(-forecast)

#Features
X = np.array(frame.drop(['Prediction'],1))
X = preprocessing.scale(X)

X_forecast = X[-forecast:]
X = X[:-forecast]
#Labels
y = np.array(frame[['Prediction']])
y = y[:-forecast]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

clf = LinearRegression()
clf.fit(X_train,y_train)

confidence = clf.score(X_test, y_test)

if(confidence<0.6):
    print("Linear Regression Model will not provide accurate results,\n Confidence level is: "+confidence)
forecast_predicted = clf.predict(X_forecast)

dates = pd.date_range(start='2018-03-28',end='2018-04-26')

plt.plot(dates,forecast_predicted,color='m')
plt.grid(color='r',linestyle='-', linewidth=0.5, alpha = True)
frame['Adj. Close'].plot(color='g')

plt.show()

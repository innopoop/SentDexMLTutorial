#!/usr/bin/python
import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, model_selection, svm #support vector machine
#cross validation
#Compares different machine learning methods and get an indication of how well they work
#Estimate the parameters for the machine learning methods ==> estimate the shape of the curve ==> Training the algorithm
#Evaluate how well the machine learning methods ==> Testing the algorithm
#Do not use all the data to train the algorithm because you don't have any data to test leftover
#75% of the data for training ==> Four-fold Cross Validation
#Leave one out cross validation (every individual is a test block)
#Common: Ten-fold Cross Validataion
from sklearn.linear_model import LinearRegression
quandl.ApiConfig.api_key = "pxobq7wfiKo1FWAfv4-M"

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df [['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]


#The last Adj. Close is a feature: measurable category that may be independent
#attributes that may cause the adjusted close price in 10 days to change (10%, not exactly 10 days)
forecast_col = 'Adj. Close' #you can use linear regression on things other than stocks

#fill NaN with a value-- treated as an outlier
df.fillna(-99999, inplace=True)

#regression used to forecast out
#predict the price of today from 30 days ago
forecast_out = int(math.ceil(0.01*len(df)))
#THIRTYY DAYS AGO

#create LABEL: this shifts the columns to the UP (negatively)
#adjusted close price of 10 days ago
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'],1)) #features are everything besides your label column
y = np.array(df['label']) #labels

X = preprocessing.scale(X) #standardize the x variables
#scale them alongside your other values
#don't do this for high frequency data analysis

#X = X[:-forecast_out+1]
print(len(X),len(y))

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2)
#20% of the dat should be used as test data
#shuffles the data and stores them

# clf = LinearRegression() #(n_jobs=-1) ==> run highest amount of threads possible
#How many jobs/threads can you run at any time? Running linearly; but if larger, it'll run training a lot faster
# clf.fit(X_train, y_train)
# accuracy = clf.score(X_test, y_test) #confidence means something else
#Accuracy == squared error

#Use Support Vector Machines Instead
clf = svm.SVR(gamma = "auto")
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

#look at the documentation for the algorithms

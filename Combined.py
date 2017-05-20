# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:09:17 2017

@author: biank
"""


from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot as plt
from pandas.tools.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.svm import SVR

def parser(x):
    return datetime.strptime(x,'%Y-%m-%d')
file = open('Hybrid.txt','w')
companies = ['PM','T','KODK','JPM','GM','GE','C','ANAT']
errorlist=[]
for k in range(len(companies)) :
    series = read_csv('/home/bijie/Desktop/stockData/'+companies[k]+'.csv',sep=',',usecols=[0,4], parse_dates=[0], header=0, index_col=0, squeeze=True, date_parser = parser)

    series = np.array(series)
    series = series[0:100]
    epsilon = 0.2# if the difference between predicted value and true value>epsilon in ARIMA, feed into SVR
    outlier = list()

    size = int(len(series) * 1)
    train, test = series[0:size], series[size:len(series)]
    true_data = list()
    history = list()
    start_point = 10
    for i in range(0,start_point):
        history.append(train[i])
    predictions = list()
    for t in range(start_point,len(train)):
        model = ARIMA(history, order=(0,1,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        if abs(yhat-train[t]) > epsilon:
            outlier.append(train[t])#if a point is outlier, put its index into the set
        else:
            predictions.append(yhat)
            true_data.append(train[t])
            obs = train[t]
            history.append(obs)
          #  print('predicted=%f, expected=%f' % (yhat, obs))


    history = list()
    history.append(outlier[0])
    for t in range(1, len(outlier)):
        predicted_data = np.arange(len(history),len(history)+1)
        predicted_data = np.expand_dims(predicted_data,axis=1)
        train_data = np.arange(0,len(history))
        train_data = np.expand_dims(train_data,axis=1)
        svr = SVR(kernel='rbf', C=1e3, gamma = 1/1250)
        yhat = svr.fit(train_data,history).predict(predicted_data)
        true_data.append(outlier[t])
        predictions.append(yhat)
        obs = outlier[t]
        history.append(obs)
      #  print('predicted=%f, expected=%f' % (yhat, obs))
    #========================================
    error = mean_squared_error(true_data, predictions)
    print('Test MSE: %.3f' % error)
    file.write('Hybrid Model'+'\n'+companies[k])
    file.write('Test MSE: %.3f' % error)
    # plot

    plt.plot(true_data, color='blue')
    plt.plot(predictions, color='red')
    plt.suptitle('Hybrid'+'\n'+companies[k])

    plt.show()
file.close()

# -*- coding: utf-8 -*-
"""
Created on Thu May 18 00:58:08 2017

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
file = open('SVR.txt','w')
companies = ['PM','T','KODK','JPM','GM','GE','C','ANAT']
errorlist=[]
for k in range(len(companies)) :
    series = read_csv('/home/bijie/Desktop/stockData/'+companies[k]+'.csv',sep=',',usecols=[0,4], parse_dates=[0], header=0, index_col=0, squeeze=True, date_parser = parser)

    series = np.array(series)
    """
    series_train = series[0:250]
    series_test = series[250:255]

    train_data = np.arange(0,250)
    train_data = np.expand_dims(train_data,axis=1)
    predicted_train = series_train

    test_data = np.arange(250,255)
    test_data = np.expand_dims(test_data,axis=1)
    true_test = series_test


    svr = SVR(kernel='rbf',C=1e3,gamma=0.1)
    predicted_test = svr.fit(train_data,predicted_train).predict(test_data)
    #plot
    lw = 2
    plt.scatter(train_data,predicted_train,color='pink',label='train data')
    plt.hold('on')
    plt.plot(test_data, true_test, color='blue',label='true test data')
    plt.plot(test_data, predicted_test, color='yellow',lw=lw, label='Gaussian kernel')
    plt.xlabel('data')
    plt.ylabel('predicted value')
    plt.title('SVR Model')
    plt.legend()
    plt.show()
    """
    #size = int(len(series) * 0.995)
    size = 100;
    #train, test = series[0:size], series[size:len(series)]
    train = series[0:10]
    test = series[11:size]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        test_data = np.arange(len(history),len(history)+1)
        test_data = np.expand_dims(test_data,axis=1)
        train_data = np.arange(0,len(history))
        train_data = np.expand_dims(train_data,axis=1)
        svr = SVR(kernel='rbf', C=1e3, gamma = 1/1250)
        yhat = svr.fit(train_data,history).predict(test_data)
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    file.write('SVR'+'\n'+companies[k])
    file.write('Test MSE: %.3f' % error)

    plt.plot(test, color='blue')
    plt.plot(predictions, color='red')
    plt.suptitle('SVR'+'\n'+companies[k])

    #plt.show()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
file.close()

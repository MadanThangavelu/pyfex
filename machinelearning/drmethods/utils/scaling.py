'''
Created on Aug 9, 2011

@author: madan
'''
from __future__ import division
from numpy import min, max


def scale_mean_zero(train_data, test_data):
    ''' Scaling is based on 
    http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#f408
    Formula: x''=2(x-mi)/(Mi-mi)-1
    Mi  : Maximum value of feature i
    mi  : Minimum value of feature i
    
    The expectation is that the train_data is of dimensions dxn
    '''    
    minimums = min(train_data, axis=1)
    maximums = max(train_data, axis=1)
    minimums = [float(i) for i in minimums]
    maximums = [float(i) for i in maximums]
    rows, cols = train_data.shape
    
    for row in range(rows):
        train_data[row,:] = 2*(train_data[row,:] - minimums[row])/(maximums[row] - minimums[row]) - 1
        test_data[row,:] = 2*(test_data[row,:] - minimums[row])/(maximums[row] - minimums[row]) - 1
    return train_data, test_data
    
        
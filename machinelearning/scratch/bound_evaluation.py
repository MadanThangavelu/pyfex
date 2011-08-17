'''
Created on Aug 4, 2011

@author: madan
'''
from pylab import subplot, plot, show, hold
from matplotlib import pyplot
from numpy import where,array
from scipy.stats.kde import gaussian_kde
import pylab as P
import numpy as np
import numpy
from numpy import sqrt, log, log2


a = numpy.r_[0.1:100:2500j]
print a.shape
def sq_sum(data):
    s = 0
    for i in data:
        s += i
    return sqrt(s)
    
def sum_sq(data):
    s = 0
    for i in data:
        s += sqrt(i)
    return s/5

sq_sum_data = []
sum_sq_data = []

fig1 = P.figure(1)
fig1.show()

for i in range(len(a)-5):
    data = a[i:i+10]
    sq_sum_data.append(sq_sum(data))
    sum_sq_data.append(sum_sq(data))
    #print 'sq_sum', sq_sum(data), 'sum_sq', sum_sq(data)*1.0/2
    
    
plot(range(len(sq_sum_data)), sq_sum_data, 'r-')
plot(range(len(sum_sq_data)), sum_sq_data, 'b.')
fig1.canvas.draw()  
show()
    
    
'''
Created on Aug 2, 2011

@author: madan
'''
from numpy import random, where, matrix

#from data.glass.adapter import Adapter as IRIS
from data.optdigits.adapter import Adapter as IRIS
#from data.wine.adapter import Adapter as IRIS
from drmethods.kde.kde import KDECUB
def run_kde():
    print "-- Starting kde--"
    landsat_dataset = IRIS(crossvalidation = 0)
    test_data,test_label,train_data,train_label = landsat_dataset.load_data()
    
    print test_data.shape
    print train_data.shape
    
    lower_dimension = 2
    kde_cub = KDECUB(train_data, train_label, lower_dimension, test_data = test_data, test_label = test_label)
    kde_cub.train()
    #kde_cub._initialize_A()
    #kde_cub._project_data()
    #kde_cub.cost()
    #kde_cub.multiprocess_gradient()
    #kde_cub.evaluate_calculation_speed()
    
    #kde_cub.test_gradient()
    
if __name__ == "__main__":
    run_kde()
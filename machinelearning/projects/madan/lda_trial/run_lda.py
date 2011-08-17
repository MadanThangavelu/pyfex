'''
Created on Aug 19, 2010

@author: Madan Thangavelu
'''
from data.optdigits.adapter import Adapter as IRIS
from drmethods.lda.method import lda as LDA
from drmethods.pca.method import pca as PCA
from numpy import transpose,dot
from numpy import float32,zeros
from plotting.plot2d.plotter import plot_2d
from pylab import figure
from plotting.plot2d.plotter import plot_2d
from classifiers.utils.probability_of_error import calculate_probability_of_error
import mlpy
from numpy import array
from numpy import int,nonzero
import numpy as np

def trial_running_lda():
    iris_dataset = IRIS(crossvalidation=1)
    test_data,test_label,train_data,train_label = iris_dataset.load_data()

    run_lda = 0
    run_pca = 1
    
    if run_lda:
        lda_trained_engine = LDA(train_data,train_label,dimension=2)
        dimension_reduced_test_data = lda_trained_engine(test_data)
        dimension_reduced_train_data = lda_trained_engine(train_data)

    if run_pca:
        pca = PCA(train_data,output_dim=8)
        dimension_reduced_train_data = pca.execute(train_data,n=2)
        dimension_reduced_test_data = pca.execute(test_data,n=2)

    
    knn = mlpy.Knn(k = 1)
    knn.compute(dimension_reduced_train_data, train_label)
    predictions = knn.predict(dimension_reduced_test_data)
    perr =  calculate_probability_of_error(predictions,test_label,train_label)
    print perr



    
    #markers = {'1':'r.','2':'g.','3':'b.'}
    #plot_2d(dimension_reduced_data,test_label.tolist(),markers)
    #res = y.execute(test_data.T)
    #print res
    
if __name__ == "__main__":
    trial_running_lda()
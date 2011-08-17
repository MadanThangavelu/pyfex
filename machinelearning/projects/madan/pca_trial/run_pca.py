'''
Created on Jul 29, 2010

@author: Madan Thangavelu
'''
from data.optdigits.adapter import Adapter as IRIS
from drmethods.pca.method import pca as PCA
from numpy import transpose,dot
from numpy import float32,zeros
from plotting.plot2d.plotter import plot_2d
from classifiers.knn.knn import knn_train, knn_test
from pylab import figure
from numpy import random, where

def trial_running_pca():
    lower_dimension = 4
    iris_dataset = IRIS(crossvalidation=1)
    test_data,test_label,train_data,train_label = iris_dataset.load_data()
    test_points,test_dimension = test_data.shape
    train_points,train_dimension = train_data.shape
    complete_data = zeros((test_points+train_points,test_dimension),float32)
    complete_data[:test_points,:] = test_data
    complete_data[test_points:,:] = train_data
    '''
    A = LDA(complete_data.T)
    dimension_reduced_data = dot(complete_data,A)
    markers = {'1':'r.','2':'g.','3':'b.'}
    plot_2d(dimension_reduced_data,test_label.tolist()+train_label.tolist(),markers)
    '''
    pca = PCA(train_data, output_dim=lower_dimension)
    dimension_reduced_test_data = pca.execute(test_data,n=lower_dimension)
    dimension_reduced_train_data = pca.execute(train_data,n=lower_dimension)
    train_data = train_data.T
    test_data  = test_data.T
    dimension_reduced_test_data = dimension_reduced_test_data.T
    dimension_reduced_train_data = dimension_reduced_train_data.T
    classification_error(dimension_reduced_train_data, train_label, dimension_reduced_test_data, test_label )
    print train_data.shape
    print test_data.shape
    print dimension_reduced_test_data.shape    
    
    markers = {'1':'r.','2':'g.','3':'b.', '4':'b*', '5':'g*', '6':'r*','7':'rs', '8':'gs','9':'bs'}
    plot_2d(dimension_reduced_test_data,test_label.tolist(),markers)
    
    
    #PCA(complete_data)
    
def classification_error(train_data, train_label, test_data, test_label):
    knn_model   = knn_train(train_data= train_data, train_label= train_label)
    predictions = knn_test(knn_model = knn_model, test_data = test_data) 
    misclassified = len(where(predictions - test_label !=0)[0])
    total_points  = len(test_label)
    print "predictions : ", misclassified*1.0/total_points
    
if __name__ == "__main__":
    trial_running_pca()
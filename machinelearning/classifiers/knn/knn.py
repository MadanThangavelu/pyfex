from shogun.Features import RealFeatures, Labels
from shogun.Classifier import KNN
from shogun.Distance import EuclidianDistance
from numpy import array, float64, where

'''
Usage

'''
def knn_train(train_data=None, train_label = None, k=1):
    train_data  = RealFeatures(train_data)
    distance    = EuclidianDistance(train_data, train_data)
    try:
        train_label = Labels(array(train_label.tolist(), dtype=float64))
    except Exception as e:
        print e
        raise Exception
    knn_model   = KNN(k, distance, train_label)
    knn_train   = knn_model.train()
    return knn_model

def knn_test(knn_model = None, test_data=None):
    test_data=RealFeatures(test_data)
    output=knn_model.classify(test_data).get_labels()
    return output

def knn_classification_error(train_data, train_label, test_data, test_label):
    ''' Not working.. a working function is in run_pca, run_lda modules '''
    knn_model            = knn_train(train_data= train_data, train_label= train_label)
    predictions          = knn_test(knn_model = knn_model, test_data = test_data) 
    misclassified_points = len(where(predictions - test_label !=0)[0])
    total_points         = len(test_label)
    error_percentage     = misclassified_points*1.0/total_points
    return misclassified_points, total_points, error_percentage
        
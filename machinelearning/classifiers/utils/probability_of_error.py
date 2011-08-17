'''
Created on Aug 25, 2010

@author: Madan Thangavelu
'''

from numpy import nonzero

def calculate_probability_of_error(predictions, test_labels,train_labels):
    test_label_dict = {}
    test_total_points = 0
    for test_label in test_labels:
        test_total_points += 1
        if test_label_dict.has_key(test_label):
            test_label_dict[test_label] += 1
        else:
            test_label_dict[test_label] = 1
            
    for key in test_label_dict.keys():
        test_label_dict[key] = 1.0*test_label_dict[key]/test_total_points # probability is calculated for each class
    
    
    train_label_dict = {}
    train_total_points = 0
    for train_label in train_labels:
        train_total_points += 1
        if train_label_dict.has_key(train_label):
            train_label_dict[train_label] += 1
        else:
            train_label_dict[train_label] = 1
            
    for key in train_label_dict.keys():
        train_label_dict[key] = 1.0*train_label_dict[key]/train_total_points # probability is calculated for each class
    
    
    
    classes = test_label_dict.keys()
    probability_error = 0
    for _class in classes:
        class_indexes = nonzero(test_labels == _class)[0]
        error_count = len(nonzero(predictions[class_indexes] - test_labels[class_indexes])[0])
        probability_error += train_label_dict[_class]*error_count/len(class_indexes)
    return probability_error
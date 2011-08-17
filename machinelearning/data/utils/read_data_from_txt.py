'''
Created on Jul 29, 2010

@author: Madan Thangavelu
'''

from numpy.random import shuffle
from numpy import savez,array,int

def split_save_train_test(data_label,train_percent,test_percent,save_file_name,save_files=True):
    n,d = data_label.shape

        
    shuffle(data_label) #Inplace shuffling as in data variable itself gets changes even in the calling function
    labels = data_label[:,-1]
    data   = data_label[:,:-1]    
    cut_off_point_for_test = int(round(n*test_percent))
    # Split the data and the labels at the cutoff point
    test_data   = array(data[:cut_off_point_for_test,:])
    test_label  = array(labels[:cut_off_point_for_test],dtype=int)
    
    train_data  = array(data[cut_off_point_for_test:,:]) 
    train_label = array(labels[cut_off_point_for_test:],dtype=int)
    if save_files:
        savez(save_file_name,\
              test_data  = test_data,
              test_label = test_label,\
              train_data = train_data,\
              train_label= train_label)
    
    return test_data,test_label,train_data,train_label

    
    
    
    
    
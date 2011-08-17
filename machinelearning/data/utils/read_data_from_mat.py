'''
Created on Aug 15, 2010

@author: Deepthi
'''
from scipy.io import loadmat
from numpy import ones,array,int32,float32, float64
from config.base import Configuration
from scipy.io import loadmat

def load_data_from_mat(directory_path,no_classes,crossvalidation=0, scale=True):
    '''
        Loads train and test data from the given directory_path and 
        file names as , test1,test2..test_n and train1,train2,.. train_n
    '''
    test_data = []
    test_label = []
    train_data = []
    train_label =[]
    if crossvalidation != 0:
        crosssetup_file = directory_path + '/cross_setup'+str(crossvalidation)+'.mat'
        crosssetup_dict = loadmat(crosssetup_file,appendmat=True)
        cross_setup =  crosssetup_dict['cross_setup'+str(crossvalidation)]
    
        
    for i in range(1,no_classes+1):
        test_filename =  directory_path+"/test"+str(i) 
        train_filename = directory_path+"/train"+str(i)
        raw_data_test = loadmat(test_filename,appendmat=True)
        raw_data_train = loadmat(train_filename,appendmat=True)
        
        test_key = "test"+str(i)
        train_key = "train"+str(i)
        
        if crossvalidation != 0:
            all_points = raw_data_train[train_key].tolist() + raw_data_test[test_key].tolist()
            all_points = array(all_points,dtype=float32)
            train_crosssetup = array(cross_setup[i-1,0][0]) - 1 # subtract 1 for matlab to python indexing
            test_crosssetup = array(cross_setup[i-1,1][0]) - 1 # subtract 1 for matlab to python indexing
            raw_data_train[train_key] = all_points[train_crosssetup,:]
            raw_data_test[test_key] = all_points[test_crosssetup,:]
        
        test_n,test_d = raw_data_test[test_key].shape
        train_n,train_d = raw_data_train[train_key].shape
        
        test_labels = i*ones(test_n,dtype=int32)
        train_labels = i*ones(train_n,dtype=int32)
        
        test_label = test_label + test_labels.tolist()
        train_label = train_label + train_labels.tolist()
        
        test_data = test_data + raw_data_test[test_key].tolist()
        train_data = train_data + raw_data_train[train_key].tolist()  
    
    tmp_test_data = array(test_data , dtype=float64) # Float 64 helps shogun
    tmp_train_data = array(train_data , dtype=float64) # Float 64 helps shogun
    tmp_test_label = array(test_label, dtype=int32) # certain classifiers only work in this dtype
    tmp_train_label = array(train_label,dtype=int32) # certain classifiers only work in this dtype
             
    test_data = tmp_test_data
    train_data = tmp_train_data
    test_label = tmp_test_label
    train_label = tmp_train_label
    
    return test_data,test_label,train_data,train_label


if __name__ == "__main__":
    #directory_path = '/home/deepthi/dimension_reduction/trunk/machinelearning/data/glass/raw_dataset'
    #no_classes = 6
    directory_path = '/home/madan/development/CUB/cub_repo/trunk/machinelearning/data/iris/raw_dataset'
    no_classes = 3    
    test_data, test_label,train_data,train_label = load_data_from_mat(directory_path,no_classes)
    print test_data,test_label,train_data,train_label
    


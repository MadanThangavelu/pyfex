'''
Created on Jul 27, 2010

@author: Madan Thangavelu
'''
from data.utils.read_data_from_txt import split_save_train_test
from data.utils.read_data_from_mat import load_data_from_mat
from numpy import genfromtxt
from numpy import load
from config.base import Configuration
from scipy.io import loadmat
class Adapter(object):
    '''
    This file converts the given data file into the format
    that machinelearning project files are going to use
    '''
    def __init__(self, crossvalidation = 0):
        #self.processed_data_filename = '/home/madan/development/CUB/cub_repo/trunk/machinelearning/data/iris/processed_dataset/iris.dataset'
        self.directory_path = Configuration("madan").base_url + '/data/iris/raw_dataset'
        #self.raw_data_filename       = 'raw_dataset/iris.data'
        self.crossvalidation         = crossvalidation
    def process_data(self):
        ''' In this data, the following class conventions are used:
        Iris_setosa     = 1
        Iris_versicolor = 2
        Iris_virginica  = 3
        '''
        file = self.raw_data_filename
        fp = open(file,'r')
        data_label = genfromtxt(fname=fp,delimiter=',')
        


        train_percent    = 0.30
        test_percent     = 0.70
        split_save_train_test(data_label,train_percent,test_percent,self.processed_data_filename,save_files=True)
    
        
    def load_data(self):
        ''' When a dataset loads we will have to return the following to the program 
            test_data, test_label, train_data,train_label
        '''
        '''
        data_files  = load(self.processed_data_filename+'.npz')
        test_data   = data_files['test_data']
        test_label  = data_files['test_label']
        train_data  = data_files['train_data']
        train_label = data_files['train_label']
        '''
        no_classes = 3
        test_data,test_label,train_data,train_label  = load_data_from_mat(self.directory_path,no_classes,crossvalidation=self.crossvalidation)
        return test_data,test_label,train_data,train_label
'''
if __name__ == "__main__":
    data_reader = Adapter()
    data_reader.process_data()
    #test_data,test_label,train_data,train_label = data_reader.load_data()
'''

if __name__ == "__main__":
    kk = Configuration("madan").base_url + '/data/iris/raw_dataset/cross_setup1.mat'
    raw_data_test = loadmat(kk,appendmat=True)
    cross_setup =  raw_data_test['cross_setup1']
    print cross_setup[2,0][0]
    


    

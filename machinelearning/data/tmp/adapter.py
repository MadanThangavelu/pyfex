'''
Created on Aug 1, 2010

@author: Balaji
'''
'''

'''
from data.utils.read_data_from_txt import split_save_train_test
from numpy import genfromtxt
from numpy import load

class Adapter(object):
    '''
    This file converts the given data file into the format
    that machinelearning project files are going to use
    '''
    def __init__(self):
        # self.processed_data_filename = '/home/madan/development/CUB/cub_repo/trunk/machinelearning/data/iris/processed_dataset/iris.dataset' 
        self.processed_data_filename = 'C:\\Users\\Balaji\\PythonWork\\trunk\\machinelearning\\data\\tmp\\processed_dataset\\dummy.dataset'
        self.raw_data_filename       = 'C:\\Users\\Balaji\\PythonWork\\trunk\\machinelearning\\data\\tmp\\raw_dataset\\dummy.txt'
        
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
        data_files  = load(self.processed_data_filename+'.npz')
        test_data   = data_files['test_data']
        test_label  = data_files['test_label']
        train_data  = data_files['train_data']
        train_label = data_files['train_label']        
        return test_data,test_label,train_data,train_label
        
if __name__ == "__main__":
    data_reader = Adapter()
    data_reader.process_data()
    #test_data,test_label,train_data,train_label = data_reader.load_data()
    


    

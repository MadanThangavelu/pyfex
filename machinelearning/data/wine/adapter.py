from data.utils.read_data_from_mat import load_data_from_mat
from config.base import Configuration

class Adapter(object):
    '''
    This file converts the given data file into the format
    that machinelearning project files are going to use
    '''
    def __init__(self, crossvalidation = 0):
        self.directory_path = Configuration("madan").base_url + '/data/wine/raw_dataset'
        self.crossvalidation         = crossvalidation
    def process_data(self):
        ''' Data comes from mat files and hence no processing required
        '''
        pass
    
    def load_data(self):
        ''' When a dataset loads we will have to return the following to the program 
            test_data, test_label, train_data,train_label
        '''
        no_classes = 3
        test_data,test_label,train_data,train_label  = load_data_from_mat(self.directory_path,no_classes,crossvalidation=self.crossvalidation)

        
        return test_data,test_label,train_data,train_label
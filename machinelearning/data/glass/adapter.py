from data.utils.read_data_from_mat import load_data_from_mat
from config.base import Configuration
from scipy.io import loadmat
class Adapter(object):
    '''
    This file converts the given data file into the format
    that machinelearning project files are going to use
    '''
    def __init__(self):
        self.directory_path = Configuration("madan").base_url + '/data/glass/raw_dataset'
    def process_data(self):
        ''' Data comes from mat files and hence no processing required
        '''
        pass
    
    def load_data(self):
        ''' When a dataset loads we will have to return the following to the program 
            test_data, test_label, train_data,train_label
        '''
        no_classes = 5
        test_data,test_label,train_data,train_label  = load_data_from_mat(self.directory_path,no_classes)     
        return test_data,test_label,train_data,train_label


'''
Created on Aug 15, 2010

@author: Madan Thangavelu
'''
from data.landsat.adapter import Adapter as LANDSAT

def run_nca():
    print "-- Starting NCA --"
    landsat_dataset = LANDSAT()
    test_data,test_label,train_data,train_label = landsat_dataset.load_data()
    print test_data.shape
    print train_data.shape

if __name__ == "__main__":
    run_nca()
'''
Created on Aug 15, 2010

@author: Madan Thangavelu
'''
from numpy import nonzero
from numpy import random
from numpy import dot,exp,zeros,float
from multiprocessing import Process

def update_numerator_matrix(A,data,label,numerator_matrix,start,end):
    for idx in range(start,end):
        data_class_id = label[idx]
        same_class_point_idx = nonzero(label==data_class_id)[0]
        for same_class_point in same_class_point_idx:
            point = data[same_class_point,:]
            x_i  = data[idx,:]
            x_j  = point
            Ax_i = dot(A,x_i)
            Ax_j = dot(A,x_j)
            Ax_i_minus_Ax_j = Ax_i - Ax_j
            numerator_matrix[idx,same_class_point] = exp(-dot(Ax_i_minus_Ax_j,Ax_i_minus_Ax_j.T))

def nca(data, label):
    n,d = data.shape
    A = random.rand(2,d)
    numerator_matrix = zeros((n,n),dtype=float)
    start = 0
    end =   n
    update_numerator_matrix(A,data,label,numerator_matrix,start,end)
    print numerator_matrix[0,0]
    print numerator_matrix[0,1]
    print numerator_matrix[0,2]
    print "sum", sum(numerator_matrix)
    print numerator_matrix.shape
    print numerator_matrix[1,0]
    
if __name__ == "__main__":
    from data.iris.adapter import Adapter as LANDSAT
    print "-- Starting NCA --"
    landsat_dataset = LANDSAT()
    test_data,test_label,train_data,train_label = landsat_dataset.load_data()
    nca(train_data,train_label)
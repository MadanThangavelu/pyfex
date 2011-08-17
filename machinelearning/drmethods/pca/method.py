'''
Created on Jul 29, 2010

@author: Madan Thangavelu
'''
#from pca_module import PCA_nipals as PCA_svd
from mdp.nodes import PCANode

def pca(data,output_dim=2, standardize=True):
    ''' This package is using the PCA module form 
        url : http://folk.uio.no/henninri/pca_module/
    '''
    '''
    T, P, explained_var = PCA_svd(data, standardize=True)
    return T,P,explained_var
    '''
    
    ''' ALTERNATE MPLEMENTATION FROM MDP '''
    pca_node = PCANode(output_dim=output_dim)
    pca_node.train(data)
    pca_node.stop_training()
    return pca_node   
    

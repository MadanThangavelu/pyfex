'''
Created on Aug 19, 2010

@author: Madan Thangavelu
'''

from mdp.nodes import FDANode


def lda(train_data,train_label,dimension=2):
    ''' This package is using the FDA module from
        mdp package (modular toolkit for data processing
        http://mdp-toolkit.sourceforge.net/
    '''
    lda_node = FDANode(output_dim=dimension)
    
    lda_node.train(train_data,train_label)
    lda_node.stop_training()
    lda_node.train(train_data,train_label)
    lda_node.stop_training()
    return lda_node

'''
Created on Aug 19, 2010

@author: Madan Thangavelu
'''

class Configuration(object):
    def __init__(self,developer):
        if developer == "madan":
            self.base_url = "/home/madan/development/CUB/cub_repo/trunk/machinelearning"
        elif developer == "deepthi":
            self.base_url = "/home/deepthi/dimension_reduction/trunk/machinelearning"
        else:
            raise ("Developer unable to find. Add your name to config.base.py")

    
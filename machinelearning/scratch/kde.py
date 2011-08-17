'''
Created on Aug 4, 2011

@author: madan
'''
from ctypes import ARRAY

"""
Created on Sun Jun 19 20:32:51 2011

@author: endolith@gmail.com

Example of 2D kernel density estimation in Python.  

Modification of the example in http://mail.scipy.org/pipermail/scipy-user/2008-July/017603.html to fix the axes and show the scatter plot on top of the KDE image.

Example image at http://flic.kr/p/9V6onm
"""

import numpy as np
import scipy.stats as stats
from matplotlib.pyplot import imshow, scatter, show


# Create some dummy data
rvs = np.append(stats.norm.rvs(loc=2,scale=1,size=(200,1)),
                stats.norm.rvs(loc=1,scale=3,size=(200,1)),
                axis=1)

kde = stats.kde.gaussian_kde(rvs.T)

# Regular grid to evaluate kde upon
x_flat = np.r_[rvs[:,0].min():rvs[:,0].max():128j]
y_flat = np.r_[rvs[:,1].min():rvs[:,1].max():128j]
x,y = np.meshgrid(x_flat,y_flat)
grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)

z = kde(grid_coords.T)
z = z.reshape(128,128)

scatter(rvs[:,0],rvs[:,1],alpha=0.5,color='white')
imshow(z,aspect=x_flat.ptp()/y_flat.ptp(),origin='lower',extent=(rvs[:,0].min(),rvs[:,0].max(),rvs[:,1].min(),rvs[:,1].max()))
show()

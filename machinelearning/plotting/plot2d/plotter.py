'''
Created on Jul 29, 2010

@author: Madan Thangavelu
'''
from pylab import subplot, plot, show
from matplotlib import pyplot
from numpy import where,array
from scipy.stats.kde import gaussian_kde
import pylab as P
import numpy as np

fig1 = P.figure(1)
fig1.show()
fig1.canvas.draw()
def plot_2d(two_dimension_data,label,markers):
    ''' data format is nxd
    '''
    P.clf()
    array_label = array(label) 
    # label has to be converted back from list to numpy as where() can only work on numpy arrays
    
    for key in markers.keys():
        plot_marker = markers[key]
        indexes = where(array_label==int(key))
        x = two_dimension_data[:,indexes[0]]
        P.plot(x[0,:],x[1,:],plot_marker,ms=9)
    fig1.canvas.draw()

def plot_2d_kde(points, covariance_2d):
    points = points.T
    ld_gaussian_kde = gaussian_kde(points.T)
    ld_gaussian_kde.covariance = covariance_2d
    
    # Regular grid to evaluate kde upon
    x_flat = np.r_[points[:,0].min():points[:,0].max():128j]
    y_flat = np.r_[points[:,1].min():points[:,1].max():128j]
    x,y = np.meshgrid(x_flat,y_flat)
    grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)
    
    z = ld_gaussian_kde(grid_coords.T)
    z = z.reshape(128,128)
    
    pyplot.scatter(points[:,0],points[:,1],alpha=0.5,color='white')
    pyplot.imshow(z,aspect=x_flat.ptp()/y_flat.ptp(),origin='lower',extent=(points[:,0].min(),points[:,0].max(),points[:,1].min(),points[:,1].max()))
    #pyplot.show()
    fig1.canvas.draw()
    print "Plotted something"

if __name__ == "__main__":
    a = array([[1,2],
               [1.2,2.1],
               [5,6]])
    print a.shape
    label = [1,1,2]
    marker = {'1':'r.','2':'b.'}
    plot_2d(a,label,marker)

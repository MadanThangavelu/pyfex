'''
Created on Aug 2, 2011

@author: madan
'''
from numpy import random, where, matrix, exp, zeros, multiply, ones, sum, array
from drmethods.utils.scaling import scale_mean_zero
from scipy.stats.kde import gaussian_kde
from plotting.plot2d.plotter import plot_2d, plot_2d_kde
import numpy
from classifiers.knn.knn import knn_train, knn_test
from multiprocessing import pool
from drmethods.pca.method import pca
from scipy import random, linalg
from multiprocessing import Pool, Lock

POOL_SIZE = 10
BATCH_SIZE = 1000

def multiprocessing_fast_chernoff_distance_matrix(required_data):
    ''' Calculates the chernoff distance between points 
    in class_i and class_j
    '''
    class_i, class_j, ld_training_data, class_indexes,\
        ld_sigma_inv  = required_data
    #print "calculating ", class_i, class_j
    d_ch = 0
    points_class_i =  ld_training_data[:, class_indexes[class_i]]
    points_class_j =  ld_training_data[:, class_indexes[class_j]]
    rows_i, columns_i = points_class_i.shape
    rows_j, columns_j = points_class_j.shape
    dimen_i = rows_i
    nopts_cl_i = columns_i
    dimen_j = rows_j
    nopts_cl_j = columns_j
    
    X = points_class_i
    Y = points_class_j
   
    tmp1 = multiply(X,(ld_sigma_inv*X)).T*ones((dimen_i, nopts_cl_j))
    tmp2 = 2*(X.T*ld_sigma_inv*Y)
    tmp3 = ones((nopts_cl_i, dimen_i))*multiply(Y,(ld_sigma_inv*Y))
    c_dist = 1.0/8*(tmp1 - tmp2 + tmp3)
    distance_matrix = exp(-c_dist)
    return distance_matrix
    

def multiprocessing_fast_chernoff_distance(required_data):
    distance_matrix = multiprocessing_fast_chernoff_distance_matrix(required_data)
    #return float(sum(sum(distance_matrix, axis=1), axis=0))
    return distance_matrix
    
    
    
def multiprocessing_fast_chernoff_gradient(required_data):
    class_i, class_j, A, sigma, ld_sigma_inv, training_data, ld_training_data, \
            class_indexes, pre_calculated_chernoff_distance_matrix = required_data
    
    points_class_i_z  = training_data[:, class_indexes[class_i]]
    points_class_j_z  = training_data[:, class_indexes[class_j]]
    points_class_i    = ld_training_data[:, class_indexes[class_i]]
    points_class_j    = ld_training_data[:, class_indexes[class_j]]
    rows_i, columns_i = points_class_i.shape
    rows_j, columns_j = points_class_j.shape
    X = points_class_i_z
    Y = points_class_j_z
    nopts_x = points_class_i_z.shape[1]
    nopts_y = points_class_j_z.shape[1]
    attrib = points_class_i_z.shape[0] # sames as rows_j_z
    distance_matrix = pre_calculated_chernoff_distance_matrix[class_i, class_j]
    phi = distance_matrix
    '''
    psi1 = phi* ones(nopts_y,1);
    psi2 = phi'* ones(nopts_x,1);
    M = inv(A * cv_sum_hd{i,j}* A');
    '''       
    psi1 = phi*ones((nopts_y, 1)) 
    psi2 = phi.T*ones((nopts_x, 1))
    M    = ld_sigma_inv
    
    '''
    % Gradient term 1
    tmp1 = (X .* (ones(attrib,1)*psi1')) * X';
    tmp2 = X*phi*Y';
    tmp3 = Y*phi'*X';
    tmp4 = (Y .* (ones(attrib,1)*psi2')) * Y';
    t1 = tmp1 - tmp2 - tmp3 + tmp4;
    gTerm1 = M*A*t1;
    '''
    tmp1 = multiply(X, (ones((attrib,1))*psi1.T))*X.T
    tmp2 = X*phi*Y.T
    tmp3 = Y*phi.T*X.T
    tmp4 = multiply(Y, (ones((attrib,1))*psi2.T))*Y.T
    t1 = tmp1 - tmp2 - tmp3 + tmp4
    gTerm1 = M*A*t1
    
    '''
    % Gradient term 2
    Myi = cv_hd{i};
    Myj = cv_hd{j};
    gTerm2 = gTerm1*A'*(M*A)*(s*Myi + (1-s)*Myj);
    '''
    gTerm2 = gTerm1*A.T*(ld_sigma_inv*A)*sigma
    
    return (1.0/4)*(gTerm1 - gTerm2)





class KDECUB(object):
    def __init__(self, training_data, train_label, lower_dimension, test_data = None, test_label = None, use_multiprocessing=True, scale_data = True):
        
        self.training_data    = matrix(training_data).T # matrix(0.001*random.randn(training_data.T.shape))# d x n
        self.training_data    = self.training_data + matrix(0.0001*random.randn(*self.training_data.shape))
        if test_data != None:
            self.test_data    = matrix(test_data).T
            self.test_label  = test_label
            
        if scale_data:
            self.training_data, self.test_data = scale_mean_zero(self.training_data, self.test_data)
                    
        self.lower_dimension  = lower_dimension
        self.sigma            = matrix(self.estimate_bandwidth().covariance)
        self.train_label      = train_label
        self.classes          = sorted(list(set(self.train_label)))
        self.class_indexes    = self._prepare_class_indexes()
        self.ld_training_data = None
        self.ld_sigma         = None
        self.d_ch_ij_cache    = {}
        
        self.pre_calculated_chernoff_distance_matrix = {}
        
        #Processing speed
        self.use_multiprocessing = use_multiprocessing
        marker_points = ['r.','g.','b.','c.','y^','mo','ko','r*','ko','yo','ro','bo','g^','k^','m^','r^',\
                         'b^','c*','rs','ks','rd','bd','gd','kd']

        marker_class_numbers = range(1, len(marker_points))
        self.markers =dict(zip(marker_class_numbers, marker_points))            
            
        cost_calculation_lock = Lock()
        gradient_calculation_lock = Lock()
        
    def test_gradient(self):
        self._initialize_A()
        self._project_data()
        self._clear_d_ch_ij_cache()
        
        cost_1     = self.cost()
        gradient_1 = self.gradient()
        eps = 1e-6
        increment = matrix(zeros(self.A.shape))
        increment[0,0] = eps
        self.A += increment

        #self._initialize_A()
        self._project_data()
        self._clear_d_ch_ij_cache()
        
        cost_2   = self.cost()
        
        print gradient_1
        print "C1", cost_1
        print "C2", cost_2
        print "Difference : ", 1.0*(cost_2 - cost_1)/eps
        print "Should equal :", gradient_1[0,0]
        
        
    def _clear_d_ch_ij_cache(self):
        self.d_ch_ij_cache    = {}
        
    def _clear_distance_cache(self):
        self.pre_calculated_chernoff_distance_matrix = {}
        
    def _initialize_A(self):
        ''' Random initialization '''
        attributes = self.training_data.shape[0]
        self.A = matrix(random.randn(self.lower_dimension, attributes))
        
        ''' Initialize with pca plane '''
        #from drmethods.pca.method import pca as PCA
        #pca = PCA(array(self.training_data).T, output_dim=self.lower_dimension)
        #self.A = pca.v.T
        
        
    def _project_data(self):
        self.ld_training_data = self.A*self.training_data
        self.ld_test_data     = self.A*self.test_data
        self.ld_sigma         = self.A*self.sigma*self.A.T
        self.ld_sigma_inv     = self.A*self.sigma*self.A.T
        self.ld_sigma_inv     = self.ld_sigma_inv.I
        
        # Testing bandwidth in the lower dimension
        #ld_gaussian = gaussian_kde(self.ld_training_data)
        #print ld_gaussian.covariance
        #self.plot_kde_data()
    
    def plot_kde_data(self):
        import numpy
        plot_data = numpy.array(self.ld_training_data.tolist())
        plot_2d_kde(plot_data, numpy.array(self.ld_sigma))
        
    def _prepare_class_indexes(self):
        class_indexes = {}
        for each_class in self.classes:
            class_indexes[each_class] = where(self.train_label == each_class)[0]
        return class_indexes    
    
    def estimate_bandwidth(self):
        return gaussian_kde(self.training_data)
    
    
    '''''''''''''''START'''''''''''''''''''''''''''''''''''''''''
    ''' Elaborate method of calculating the chernoff criteria '''
    def dch(self,xi,xj):
        return float(exp(-(1.0/8)*(xi - xj).T*self.ld_sigma_inv*(xi - xj)))
    
    def slow_chernoff_distance_matrix(self, class_i, class_j):
        ''' Calculates the chernoff distance matrix in slow technique
        '''
        d_ch = 0
        points_class_i = self.ld_training_data[:,self.class_indexes[class_i]]
        points_class_j = self.ld_training_data[:,self.class_indexes[class_j]]
        rows_i, columns_i = points_class_i.shape
        rows_j, columns_j = points_class_j.shape
        distance_matrix = matrix(zeros((columns_i,columns_j )))
        for i in range(columns_i):
            for j in range(columns_j):
                xi = points_class_i[:, i]
                xj = points_class_j[:, j]
                #d_ch += self.dch(xi, xj)
                distance_matrix[i, j] = self.dch(xi, xj)
        return distance_matrix
    
    def chernoff_distance(self, class_i, class_j):
        ''' Calculates the chernoff distance between points 
        in class_i and class_j
        '''
        #print "calculating ", class_i, class_j
        d_ch = 0
        points_class_i = self.ld_training_data[:,self.class_indexes[class_i]]
        points_class_j = self.ld_training_data[:,self.class_indexes[class_j]]
        rows_i, columns_i = points_class_i.shape
        rows_j, columns_j = points_class_j.shape
        
        for i in range(columns_i):
            for j in range(columns_j):
                xi = points_class_i[:, i]
                xj = points_class_j[:, j]
                d_ch += self.dch(xi, xj)
        return d_ch
    
    def chernoff_gradient(self, class_i, class_j):
        ''' Calculates the chernoff gradient between points 
        in class_i and class_j
        '''
        points_class_i_z  = self.training_data[:,self.class_indexes[class_i]]
        points_class_j_z  = self.training_data[:,self.class_indexes[class_j]]
        points_class_i    = self.ld_training_data[:,self.class_indexes[class_i]]
        points_class_j    = self.ld_training_data[:,self.class_indexes[class_j]]
        rows_i, columns_i = points_class_i.shape
        rows_j, columns_j = points_class_j.shape
        
        intermediate_gradient = zeros(self.A.shape)
        for i in range(columns_i):
            for j in range(columns_j):
                xi_z = points_class_i_z[:, i]
                xj_z = points_class_j_z[:, j]
                xi   = points_class_i[:, i]
                xj   = points_class_j[:, j]
                term1 = self.ld_sigma_inv*(xi-xj)*(xi_z - xj_z).T
                term2 = self.ld_sigma_inv*(xi-xj)*(xi - xj).T*self.ld_sigma_inv*self.A*self.sigma
                intermediate_gradient += self.dch(xi, xj) * (1.0/4) * (term1 - term2)
        return intermediate_gradient
    
    def elaborate_cost(self):
        chernoff_distance = 0
        for class_i in self.class_indexes.keys():
            for class_j in self.class_indexes.keys():
                if class_i == class_j:
                    continue
                chernoff_distance += self.chernoff_distance(class_i, class_j)
        #print "Final chernoff distance : ", chernoff_distance
        return chernoff_distance*(1.0/self.training_data.shape[1])
    
    def elaborate_gradient(self):
        gradient = matrix(zeros(self.A.shape))
        for class_i in self.class_indexes.keys():
            for class_j in self.class_indexes.keys():
                if class_i == class_j:
                    continue
                gradient += self.chernoff_gradient(class_i, class_j)
        return -1*gradient*(1.0/self.training_data.shape[1])
    '''''''''''''''END'''''''''''''''''''''''''''''''''''''''''
    
    
    '''''''''''''''START FAST EVALUATION'''''''''''''''''''''''
    def fast_chernoff_distance_matrix(self, class_i, class_j):
        ''' Calculates the chernoff distance between points 
        in class_i and class_j
        '''
        #print "calculating ", class_i, class_j
        d_ch = 0
        points_class_i = self.ld_training_data[:,self.class_indexes[class_i]]
        points_class_j = self.ld_training_data[:,self.class_indexes[class_j]]
        rows_i, columns_i = points_class_i.shape
        rows_j, columns_j = points_class_j.shape
        dimen_i = rows_i
        nopts_cl_i = columns_i
        dimen_j = rows_j
        nopts_cl_j = columns_j
        
        X = points_class_i
        Y = points_class_j
        '''
        nopts_cl_1 = size(pts_cl1,2);
        Q3 = (s)/2 * log (det(A * cv_hd * A'));
        dist = [];
        X = pts_cl1;
        nopts_cl_2 = size(pts_cl2,2);
        ACA_hd = A * cv_sum_hd* A';
        M = inv(ACA_hd);
        Y = pts_cl2;
        tmp1 = (X.*(M*X))'*ones(dimen,nopts_cl_2);
        tmp2 = 2*(X'*M*Y);
        tmp3 = ones(nopts_cl_1,dimen)*(Y.*(M*Y));
        Q1 = s*(1-s)/2 * (tmp1 - tmp2 + tmp3);
        Q2 = 0.5 * log (det(ACA_hd));
        Q4 = (1-s)/2 * log(det(A * cv_hd * A'));
        c_dist = Q1 + Q2 - Q3 - Q4;
        Distance_matrix = exp(-c_dist);
        '''        
        tmp1 = multiply(X,(self.ld_sigma_inv*X)).T*ones((dimen_i, nopts_cl_j))
        tmp2 = 2*(X.T*self.ld_sigma_inv*Y)
        tmp3 = ones((nopts_cl_i, dimen_i))*multiply(Y,(self.ld_sigma_inv*Y))
        c_dist = 1.0/8*(tmp1 - tmp2 + tmp3)
        distance_matrix = exp(-c_dist)
        return distance_matrix
    
    def fast_chernoff_distance(self, class_i, class_j):
        distance_matrix = self.fast_chernoff_distance_matrix(class_i, class_j)
        return float(sum(sum(distance_matrix, axis=1), axis=0))
    
    def fast_chernoff_gradient(self, class_i, class_j):
        points_class_i_z  = self.training_data[:,self.class_indexes[class_i]]
        points_class_j_z  = self.training_data[:,self.class_indexes[class_j]]
        points_class_i    = self.ld_training_data[:,self.class_indexes[class_i]]
        points_class_j    = self.ld_training_data[:,self.class_indexes[class_j]]
        rows_i, columns_i = points_class_i.shape
        rows_j, columns_j = points_class_j.shape
        X = points_class_i_z
        Y = points_class_j_z
        nopts_x = points_class_i_z.shape[1]
        nopts_y = points_class_j_z.shape[1]
        attrib = points_class_i_z.shape[0] # sames as rows_j_z
        distance_matrix = self.fast_chernoff_distance_matrix(class_i, class_j)
        phi = distance_matrix
        '''
        psi1 = phi* ones(nopts_y,1);
        psi2 = phi'* ones(nopts_x,1);
        M = inv(A * cv_sum_hd{i,j}* A');
        '''       
        psi1 = phi*ones((nopts_y, 1)) 
        psi2 = phi.T*ones((nopts_x, 1))
        M    = self.ld_sigma_inv
        
        '''
        % Gradient term 1
        tmp1 = (X .* (ones(attrib,1)*psi1')) * X';
        tmp2 = X*phi*Y';
        tmp3 = Y*phi'*X';
        tmp4 = (Y .* (ones(attrib,1)*psi2')) * Y';
        t1 = tmp1 - tmp2 - tmp3 + tmp4;
        gTerm1 = M*A*t1;
        '''
        tmp1 = multiply(X, (ones((attrib,1))*psi1.T))*X.T
        tmp2 = X*phi*Y.T
        tmp3 = Y*phi.T*X.T
        tmp4 = multiply(Y, (ones((attrib,1))*psi2.T))*Y.T
        t1 = tmp1 - tmp2 - tmp3 + tmp4
        gTerm1 = M*self.A*t1
        
        '''
        % Gradient term 2
        Myi = cv_hd{i};
        Myj = cv_hd{j};
        gTerm2 = gTerm1*A'*(M*A)*(s*Myi + (1-s)*Myj);
        '''
        gTerm2 = gTerm1*self.A.T*(self.ld_sigma_inv*self.A)*self.sigma
        
        return (1.0/4)*(gTerm1 - gTerm2)
                
    def fast_cost(self):
        chernoff_distance = 0
        for class_i in self.class_indexes.keys():
            for class_j in self.class_indexes.keys():
                if class_i == class_j:
                    continue
                chernoff_distance += self.fast_chernoff_distance(class_i, class_j)
        return chernoff_distance*(1.0/self.training_data.shape[1])
    
    def fast_gradient(self):
        gradient = matrix(zeros(self.A.shape))
        for class_i in self.class_indexes.keys():
            for class_j in self.class_indexes.keys():
                if class_i == class_j:
                    continue
                gradient += self.fast_chernoff_gradient(class_i, class_j)
        return -1*gradient*(1.0/self.training_data.shape[1])
    '''''''''''''''END'''''''''''''''''''''''
    
    
    '''''''''''''''MULTIPROCSSING START'''''''''''''''''''''''
    def multiprocess_cost(self):
        ''' A multiprocessing framework to parallelize
        the cost computations of kde '''        
        class_pairs = []
        required_data = []
        for class_i in self.class_indexes.keys():
            for class_j in self.class_indexes.keys():
                if class_i == class_j:
                    continue
                
                if class_j > class_i: # Calculating only one pair of the classes           
                    class_pairs.append((class_i, class_j))
                    required_data.append((class_i, class_j, self.ld_training_data, self.class_indexes, self.ld_sigma_inv))
                    
        pool = Pool(processes=8)              # start 4 worker process
        chernoff_distance_matrices = pool.map(multiprocessing_fast_chernoff_distance, required_data)
        total_chernoff_distance = 0
        for idx, class_pair in enumerate(class_pairs):
            class_i, class_j = class_pair
            self.pre_calculated_chernoff_distance_matrix[class_i, class_j] = chernoff_distance_matrices[idx]
            self.pre_calculated_chernoff_distance_matrix[class_j, class_i] = chernoff_distance_matrices[idx] 
            total_chernoff_distance += float(sum(sum(chernoff_distance_matrices[idx], axis=1), axis=0))
        
        return 2*sum(total_chernoff_distance)*(1.0/self.training_data.shape[1])
        
    def multiprocess_gradient(self):
        ''' A multiprocessing framework to parallelize
        the gradient computations  of kde '''
        required_data = []
        for class_i in self.class_indexes.keys():
            for class_j in self.class_indexes.keys():
                if class_i == class_j:
                    continue
                
                if class_j > class_i: # Calculating only one pair of the classes           
                    required_data.append((class_i, class_j, self.A, self.sigma, self.ld_sigma_inv,\
                                        self.training_data, self.ld_training_data, self.class_indexes, \
                                        self.pre_calculated_chernoff_distance_matrix ))
        
        pool = Pool(processes=8)              # start 4 worker process
        gradients = pool.map(multiprocessing_fast_chernoff_gradient, required_data)
        #gradients = multiprocessing_fast_chernoff_gradient(required_data[0])
        total_gradient = matrix(zeros(self.A.shape))
        for gradient in gradients:
            total_gradient += gradient
        
        return -1*2*total_gradient*(1.0/self.training_data.shape[1])
    '''''''''''''''MULTIPROCSSING END'''''''''''''''''''''''
    
    
    
    
    
    def cost(self):
        #cost = self.elaborate_cost()
        #print "slow cost",  cost
        #cost = self.fast_cost()
        #print "Fast Cost", cost
        cost = self.multiprocess_cost()
        #print "Multiprocess Cost", cost
        return cost
    
    def gradient(self):
        #gradient = self.elaborate_gradient()
        #print "slow gradient",  gradient
        #gradient = self.fast_gradient()
        #print "fast gradient", gradient
        
        gradient = self.multiprocess_gradient()  
        #print gradient - new_gradient    
        return  gradient
        
        
    def backtracking_step_size(self, gradient):
        '''  Refer http://www.ee.ucla.edu/~vandenbe/236C/lectures/gradient.pdf '''
        preserve_starting_A = self.A.copy()
        
        beta  = 0.01
        alpha = 1
        
        self._project_data()
        self._clear_d_ch_ij_cache()
        current_cost = self.cost()
        f_x = current_cost    
        print gradient.shape, type(gradient)
        -gradient.T*gradient
        armijo_condition = beta*numpy.trace(-gradient.T*gradient)
        
        while True:            
            self.A = preserve_starting_A.copy() - alpha*gradient
            self._project_data()
            self._clear_d_ch_ij_cache()
            stepped_cost = self.cost()
            
            lhs = stepped_cost # f(x^k + \alpha p^k)
            rhs = f_x + alpha*armijo_condition

            print "Performing backtracking search :  lhs", lhs, "rhs", rhs
            
            if lhs > rhs:
                tou = 1.0/2
                print "f_x, stepped_cost, alpha, lhs, rhs", f_x, stepped_cost, alpha, lhs, rhs
                alpha = tou*alpha
                if alpha < 1e-5:
                    alpha = False
                    break
                
            else:
                break
            
        # Revert the state of A
        self.A = preserve_starting_A.copy()
        return alpha
                
    def evaluate_calculation_speed(self, cost_function=1, gradient_function=False):
        calcuation_times = []
        number_of_times      = 10
        from datetime import datetime
        for i in range(number_of_times):
            self._initialize_A()
            self._project_data()
            self._clear_distance_cache()
            start_time = datetime.now()
            self.cost()
            self.gradient()
            end_time   = datetime.now()
            difference = end_time - start_time 
            calcuation_times.append(difference)
            
        total_time = calcuation_times[0] - calcuation_times[0]
        for calcuation_time in calcuation_times:
            total_time += calcuation_time
        print total_time/len(calcuation_times)
        
    def train(self):
        self._initialize_A()
        for i in range(500000):
            self._project_data()
            self.classification_error()
            #plot_2d(self.ld_training_data, self.train_label, self.markers)    
            self._clear_distance_cache()
            cost = self.cost()
            print cost
            gradient = self.gradient()
            step_size = self.backtracking_step_size(gradient)
            if step_size:
                self.A = self.A - step_size*gradient
            else:
                print "Constant stepping"
                self.A = self.A - 0.01*gradient
            
            #print gradient
            
    def classification_error(self):
        knn_model   = knn_train(train_data= self.ld_training_data, train_label= self.train_label)
        predictions = knn_test(knn_model = knn_model, test_data = self.ld_test_data) 
        misclassified = len(where(predictions - self.test_label !=0)[0])
        total_points  = len(self.test_label)
        print "predictions : ", misclassified, total_points, misclassified*1.0/total_points
        

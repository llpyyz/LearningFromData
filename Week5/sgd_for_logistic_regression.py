"""
EdX Learning from Data (Caltech)
HW 5 
Author : David Schonberger
Created: 11/2/2014
Implementation of Stochastic Gradient Descent for Logistic Regression
Uses MC sim to 1) generate uniform random data from the square [-1,1] X [-1,1]
and to 2) estimate the cross-entropy error on ot of sample random data from same space
"""
import random
import numpy as np

#generate uniform random points from a rectangle
def get_random_points(a = -1, b = 1, c = -1, d = 1, numpts = 2):
    """
    Input: 
    a, b, c, d, the left, right, lower, and upper bnds
    resp for the rectangle from which to generate points
    
    numpts, the number of random pts to generate from the space

    Output: numpts points randomly chosen from [a,b] X [c, d]
    """
    ptslst = []
    for i in range(numpts):
        ptslst.append((1, random.random()*(b - a) + a, random.random()*(d - c) + c)) 
    
    return ptslst

#find slope of line passing thru pts in list    
#note: each pt in lst has a '1' prepended, as per usual notation
def get_line_slope(lst):
    """
    Input:
    lst, a list of two points
    
    Output:
    slope, the slope of the line through the two points, 
    or Inf if the points have same first coordinate
    """
    slope = np.Inf
    if lst[0][1] != lst[1][1]:
        slope = (lst[0][2] - lst[1][2])/(lst[0][1] - lst[1][1])
    return slope


#Points above line are classified as -1; below as +1
def classify_points(lst, pt_on_line, slope):
    """
    Input:
    -lst, a list of points to classify
    -pt and slope which determine the classifier line
    
    Output:
    classifications, a list of +/-1 vals for the corresponding pts
    in lst    
    """
    classifications = []
    for pt in lst:
        yhat = slope * (pt[1] - pt_on_line[1]) + pt_on_line[2]
        classifications.append((-1) ** (int(pt[2] - yhat > 0))) 

    return classifications
    
#stochastic gradient descent: make one pass through
#data set (one epoch), returning adjusted weight w
def sgd(data_set, classifications, weights, eta):
    """
    Input:
    -data_set, the set of points on which to perform sgd
    -classifications, the +/-1 classifications of each point
    -weights, the weight vector at beginning of epoch
    -eta ,the learning rate
    
    Output: 
    w, the learned weight vector
    """
    indices = range(len(data_set))
    random.shuffle(indices)
    for idx in indices:
        weights = np.subtract(weights, np.multiply(eta, logistic_reg_gradient(data_set[idx], classifications[idx], weights)))
        
    return weights

#calculate the gradient vector for logistic regression on a single pt
#(see for instance, slide 23 lec 9, Caltech LFD)
def logistic_reg_gradient(x, y, w):
    """
    Input:
    x, the data point, a numpy array
    y, the -/+1 classification of x
    w, the weight vector, a numpy array
    
    Output: gradient vector, delta_E_in
    """
    delta_E_in = list(np.multiply(-y * 1.0/(1 + np.exp(y * np.dot(w,x))) , x))
    return delta_E_in

#returns 2-norm of vectors u and v    
def two_norm(u,v):
    norm = 0.0
    for idx in range(len(u)):
        norm += (u[idx] - v[idx]) ** 2
    
    return np.sqrt(norm)
    
#calculate a set of weights for logistic regression via SGD
def run_sgd():
    """
    """
    N = 100
    eta = 0.01
    max_epochs = 2000
    tol = 0.01
    curr_epochs = 0
    curr_norm = 2 * tol
    
    weights = np.array([0,0,0]) #init
    
    #get target    
    two_pts =  get_random_points(-1, 1, -1, 1, 2) #get 2 pts
    m = get_line_slope(two_pts) #target slope
    
    #get and classify data
    data = get_random_points(-1 , 1 , -1 , 1 , N)
    classifications = classify_points(data , two_pts[0] , m)    
        
    while curr_norm > tol and curr_epochs < max_epochs:
        curr_w = np.copy(weights) #save for later
        weights = sgd(data, classifications, weights, eta)
        
        curr_norm = two_norm(weights, curr_w)
        curr_epochs += 1
        
    return [weights, curr_epochs, two_pts[0], m]

#estimate E_out by MC sim
def calc_cross_entropy_error(weights, pt, slope):
    """
    """
    E_out = 0.0
    numpts = 1000
    test_set = get_random_points(-1 , 1 , -1 , 1 , numpts)
    classifications = classify_points(test_set , pt , slope)    
    
    for idx in range(numpts):
        E_out += np.log(1 + np.exp(-1.0 * classifications[idx] * np.dot(weights, test_set[idx])))
    
    return E_out / numpts

def __main__():
    
    numruns = 100
    avg_cee = 0.0
    for idx in range(numruns):
        ret = run_sgd()
        print "\nrun", idx + 1
        print "weights:", ret[0], "num epochs:", ret[1]
        cee = calc_cross_entropy_error(ret[0], ret[2], ret[3])
        avg_cee += cee
        print "cee:", cee, "\n"
        
    print
    print "avg cee:", avg_cee / numruns
    print
    

__main__()
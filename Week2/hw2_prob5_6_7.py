"""
Caltech Learning From Data
Homework 2 - Problems 5-7
Author: David Schonberger
Created: 10/10/2014
"""

#The following experiment is repeated 1000 times:

#1. Draw a pair of points randomly from [-1,1] X [-1,1]. Find the line
#through these points. This is the target function f.

#2. Generate N 1000 points drawn random uniformly from [-1,1] X [-1,1], 
# then classify each point: 
#   --points above the line will be '+' 
#   --point below will be '-'. 

#If target f happens to be vertical, points to the left will be '+' and 
#to the right will be '-'.

#In problem 5, we use linear regression to find final hypothesis g. 
#This involves computing the weight vector w = pinv(X) * y,
#where pinv(X) is the pseudoinverse of X and y is the vector
#of classifications

#To estimate P[f(x) != g(x)], we generate a large number of random
#points in the square and calculate percentage of them on which
#f and g disagree. This is our estimated probability.

import random
import numpy

"""
Problems 5,6,7 - Linear Regression
"""
#################
### Problem 5 ###
#################
def gen_two_points(a = -1,b = 1,c = -1,d = 1):
    """
    Input: a, b, c, and d, the x and y bounds
    of a rectangle in the xy plane
    
    Output: a pair of randomly generated points in [a,b] X [c,d]
    """
    point1 = (random.random() * (b - a) + a, random.random() * (d - c) + c)
    point2 = (random.random() * (b - a) + a, random.random() * (d - c) + c)
    return [point1, point2]

def calc_target_slope(point_list):
    """
    Input: list of two points [(x1,y1), (x2,y2)]
    
    Output: slope of line through the two points
    Returns numpy.Inf if x1 == x2
    """
    slope = Inf
    x1 = point_list[0][0]
    y1 = point_list[0][1]
    x2 = point_list[1][0]
    y2 = point_list[1][1]
    
    if x1 != x2:
        slope = (y2 - y1)/(x2 - x1)
        
    return slope

def gen_training_set(num_points,a = -1, b = 1, c = -1, d = 1):
    """
    Input: a, b, c, and d, the x and y bounds
    num_points, the number of points in the training set
    
    Output: training_set, a list of randomly generated points
    drawn from [a,b] X [c,d]
    """
    training_set = []
    for i in range(0 , num_points):
        training_set.append((random.random() * (b - a) + a, random.random() * (d - c) + c))
        
    return training_set

def make_trainset_matrix(data):
    """
    Input: 
    -data, a list of N points in training set
    
    Output:
    -ts_matrix, a N by d+1 matrix, where d = number
    of features in each point in data
    """
    nrow = len(data)
    ncol = 1 + len(data[0])
    ts_matrix = ones(nrow * ncol).reshape(nrow, ncol)
    for row_idx in range(0, nrow):
        for col_idx in range(1, ncol):
            ts_matrix[row_idx, col_idx] = data[row_idx][col_idx - 1]
            
    return ts_matrix
    
    
def classify(points_list, slope, point):
    """
    Input: points_list, a list of points to classify as +/-1
     slope, the slope of the target line
     point, a point on the target line
     
     Output: classification a list of +/-1 for the corresponding 
     point in points_list
    """
    classification = []
    x1 = point[0]
    y1 = point[1]
    for p in points_list:
        if slope != Inf:
            y = slope * (p[0] - x1) + y1
            classification.append((-1) ** (int(p[1] - y < 0)))
        else:
            classification.append((-1) ** (int(p[0] > x1)))
            
    return classification

def calc_weight_vec(input_matrix, output_vec):
    """
    Input: 
    -input_matrix, a N x (d+1) matrix where d is 
    the number of features for each input vector x. 
    (Each input vector x extra component x_0 = 1)
    
    -output_vec, a N by 1 vector of y values
    
    Output: w = pseudoinverse(input_matrix) * output_vec
    """
    w = dot(linalg.pinv(input_matrix), output_vec)
    return w
    
#def calc_E_in(data_set, classifications, weight_vec):
#    """
#    Input:
#    -data_set, a list of N points
#    
#    -classifications, the list of N correct classifications of each pt
#
#    -weight_vec, the weights found in lin reg, used to
#    calculate sign(w^T * x) for each x in data_set
#    
#    Output: the fraction of points misclassified by the
#    lin reg parameters. 
#    """
#    misclassified_count = 0
#    for idx in range(0, len(data_set)):
#        data_set[idx].insert(0 , 1)
#        hyp_sign  = (-1) ** (int((dot(weight_vec, data_set[idx]) < 0)))
#        if hyp_sign != classifications[idx]:
#            misclassified_count += 1
#        
#    return misclassified_count * 1.0 / len(data_set)

def calc_E_in_2(data_set_maxtrix, classifications, weight_vec):
    """
    Input:
    -data_set_matrix, N x (d+1) array of points
    
    -classifications, the list of N correct classifications of each pt

    -weight_vec, the weights found in lin reg, used to
    calculate sign(w^T * x) for each x in data_set
    
    Output: the fraction of points misclassified by the
    lin reg parameters. 
    """    
    hyp_vec = dot(data_set_maxtrix, weight_vec)
    hyp_signs = array(map(int, hyp_vec >= 0))
    classify_signs = array(map(int, array(classifications) >= 0))
    
    return 1.0 * sum(hyp_signs != classify_signs)/len(hyp_signs)

def run_sims(numsims, size):
    targets = []
    hypotheses = []
    ts_size = size
    misclassified_percent = 0.0
    for sim in range(0, numsims):
        pts = gen_two_points()
        m = calc_target_slope(pts)
        targets.append([pts[0], m]) #save 1st pt and slope for later
        training_set = gen_training_set(ts_size)
        ts_matrix = make_trainset_matrix(training_set)
        classifications = classify(training_set, m, pts[0])
        weights = calc_weight_vec(ts_matrix, classifications)
        hypotheses.append(weights)
        misclassified_percent += calc_E_in_2(ts_matrix, classifications, weights)

    return misclassified_percent / numsims
    

nosims = 1000
size = 100
res = run_sims(nosims, size)    
print "result", res

"""
print "### begin output ###"
print ""

l = gen_two_points()
print "two points of target", l
print "###"

m = calc_target_slope(l)
print "m: ",m
print"###"

ts = gen_training_set(3)
print "training set:", ts
print "###"

ts_matrix = make_trainset_matrix(ts)
print "ts mat",ts_matrix
print "###"

cl = classify(ts, m, l[0])
print "cl", cl
print "###"

#y_vec = array([-1,1,1]).transpose()
#arr = array([1,2,3,4,5,6]).reshape(3,2)

weights = calc_weight_vec(ts_matrix, cl)
#print "input arr",arr
#print "y vec", y_vec
print "lin reg weights", weights
print "###"

#pinv1 = dot(linalg.inv(dot(arr.transpose(), arr)),  arr.transpose())
#print "pinv", pinv1
#print dot(pinv1, y_vec)

print ""
print "### end output ###"

dat = [(1,2),(3,4),(5,6)]
mat = make_trainset_matrix(dat)

dat2 = array([-2,-3,4,-5,6])
dat3 = array([1,-1,-1,1,1])
b1 = array(map(int, dat2 > 0))
b2 = array(map(int, dat3 > 0))
print b1, b2, b1 == b2, abs(b1 - b2), 1.0 * sum(b1 == b2)/len(b1)
"""

#t = (1,2,3)
#u = (5,6,7)
#print t + u
#
#l1 = [1,2,3]
#print l1
#l1.insert(0,0)
#print l1

#dat2 = array([-2,-3,4,-5,6,7]).reshape(2,3)
#dat3 = array([1,-1,-1,1,1,-1]).reshape(3,2)
#res1 = dot(dat2, dat3)
#print dat2
#print dat3
#print res1
#

#cl = [1,-1,-1,1,1,-1,1,-1,-1,-1,1,1]
#print cl
#print (array(cl) >= 0)
#
#print map(int, array(cl) >= 0), type(map(int, array(cl) >= 0))



#################
### Problem 6 ###
#################

"""
Estimte E_out
"""
#numsims = 1000
#size = 10000
#E_out_est = 0.0
#for idx in range(0, numsims):
#    ts = gen_training_set(size)
#    ts_matrix = make_trainset_matrix(ts)
#    slope = targets[idx][1]
#    pt = targets[idx][0]
#    classifications = classify(ts, slope, pt)
#    E_out_est += calc_E_in_2(ts_matrix, classifications, hypotheses[idx])
#    
#print E_out_est / numsims #about 0.049, no quite as good as E_in




#################
### Problem 7 ###
#################

def pla2(ts, ts_matrix, classification, init_w):
    
    trainingset = ts
    weights = init_w
    
    hypoth_vec = dot(ts_matrix, weights)
    hypoth_signs = array(map(int, hypoth_vec >= 0))
    classification_signs = array(map(int, array(classification) >= 0))
    misclassified_points = map(int, classification_signs != hypoth_signs)
    
    maxiters = 10000
    curriter = 0
    while sum(misclassified_points) > 0 and curriter < maxiters:
        
        #pick index of misclassified point
        randindex = random.randint(0, len(misclassified_points) - 1)
        while misclassified_points[randindex] == 0:
            randindex = random.randint(0, len(misclassified_points) - 1)
        
        #update w
        xcoord = trainingset[randindex][0]
        ycoord = trainingset[randindex][1]
        sign = classification[randindex]
        weights = weights + sign * array([1.0, xcoord, ycoord])
        
        #update misclassified
        hypoth_vec = dot(ts_matrix, weights)
        hypoth_signs = array(map(int, hypoth_vec >= 0))
        misclassified_points = map(int, classification_signs != hypoth_signs)
        
#        misclassified_points = []
#        hyp_signs = []
#        for index in range(0, numpoints):
#            #compute h(x) = sign(w^T * x)
#            curr_point = trainingset[index]
#            curr_point_vec = numpy.array([1.0, curr_point[0], curr_point[1]])
#            hypoth_sign = int((-1) ** (numpy.dot(weights, curr_point_vec) < 0))
#            hyp_signs.append(hypoth_sign)
#            
#            #if misclassified, add this to list
#            if hypoth_sign != trainingset_classification[index]:
#                misclassified_points.append(index)
                
        curriter += 1

    #return [curriter , target_slope , point1, weights]
    return curriter


### end pla2 ###

numsims = 10
size = 10
iters = 0

for sim in range(0, numsims):
    print "sim #", sim
    print "###"
    
    pts = gen_two_points()
    print "pts done; pts = ", pts
    print "###"
    
    m = calc_target_slope(pts)
    print "slope done; slope = ", m
    print "###"
    
    training_set = gen_training_set(size)
    print "ts done, ts = ", training_set
    print "###"
    
    ts_matrix = make_trainset_matrix(training_set)
    print "ts mat done, ts mat = ", ts_matrix
    print "###"
    
    classifications = classify(training_set, m, pts[0])
    print "classify of ts done, cl = ", classifications
    print "###"
    
    weights = calc_weight_vec(ts_matrix, classifications)
    print "weight cal done, w = ", weights
    print "###"
    
    #ret = pla2(training_set, ts_matrix, classifications, weights)
    iters += pla2(training_set, ts_matrix, classifications, weights)
    print "back from pla, iters", iters
    print "###"

print "avg iters to converge", 1.0 * iters / numsims



#for i = 1..1000
#  generate training set of 10 pts
#  run lin reg to find weight w
#  run pla with w is initial guess, count iters to convergence

#print avg number of iters to convergence over 1000 runs

#l1 = [1,-1,-1,1,1]
#l2 = [-1,-1,1,-1,1]
#a1 = array(map(int,array(l1) > 0))
#a2 = array(map(int,array(l2) > 0))
#print a1
#print a2
#print map(int, a1 != a2), sum(map(int, a1 != a2))

#print cl
#print (array(cl) >= 0)
#
#print map(int, array(cl) >= 0), type(map(int, array(cl) >= 0))

#import numpy as np
#print np.random.random_integers(0,10)

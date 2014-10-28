"""
Learning from Data - Coursera
Homework 1, Questions 7-10: 
Perceptron Learning Algorithm (PLA)
Author: David Schonberger
Date Created: 10/2/2014
Last Modified: 10/2/2014
"""

#Data is drawn uniformly from [-1,1] X [-1,1], the 2 by 2 square centered at 
#(0,0) in xy-plane

#For each of 1000 runs, we choose a random line passing through this region 
#(possibly on its boundary). This is the target function f.

#Then we choose N = 10 (ex. 7,8) or N = 100 (ex. 9,10) random points in the 
#above square. 

#We arbitrarily classify the points. In this case, points above the line 
#will be '+' and below will be '-'. If the line happens to be vertical, 
#points to the left will be '+' and to the right will be '-'.

#After the training set is classified, we init w = (0,0,0) and then proceed 
#with the PLA until no misclassified points remain. We keep count of the number 
#of iterations needed. 
#
#Basic PLA:
#Repeat while there are misclassified points:
#
#1. Randomly pick a misclassified point (x,y) under current hypothesis. Here
# x is the feature vector, while y is the sign under target function.
#
#2. Set w := w + y * x
#
#Note that initially every point is misclassified since
#0 * x = 0 which has no sign, so it would not match the actual sign of any
#of the N classified points. Thus we initially select a random point from
#the training set and determine our new w from that.

#When no misclassified points remain, we have a set of weights w = (w0,w1,w2) 
#that give us a final hypothesis g(x) = sign(w^T * x). For our training set, 
#g(x) has the same sign as each point in the training set.

#To estimate P[f(x) != g(x)], we generate a large number of random
#points in the square and calculate percentage of them on which
#f and g disagree. This is our estimated probability.

import random
import numpy

#a = arange(15).reshape(3,5)
#print a
#
#print a.shape
#print a.ndim
#print a.dtype.name
#print a.itemsize
#print a.size
#print type(a)
#b = array([random.randint(0,10), random.randint(0,10), random.randint(0,10)])
#print b
#print type(b)
#
#c = array([random.random(), random.random(), random.random()])
#print c
#print type(c)
#print c.dtype

def pla():
    
    #generate two random points in square
    lowbnd = -1
    upbnd = 1
    rng = upbnd - lowbnd
    point1 = (random.random() * rng + lowbnd, random.random() * rng + lowbnd)
    point2 = (random.random() * rng + lowbnd, random.random() * rng + lowbnd)

    #generate slope of target function
    target_slope = numpy.Inf
    if point1[0] != point2[0]:
        target_slope = (point1[1] - point2[1])/(point1[0] - point2[0])


    trainingset = []
    numpoints = 100
    for i in range(0,numpoints):
        trainingset.append((random.random() * rng + lowbnd, random.random() * rng + lowbnd))

    #Classify the chosen points
    #Use point-slope form of line and point1 from above.
    #If point lies on or above target line, classify as +1, else -1.
    trainingset_classification = []
    for point in trainingset:
        if target_slope != numpy.Inf and target_slope != -numpy.Inf:
            y_line = target_slope * (point[0] - point1[0]) + point1[1]
            trainingset_classification.append((-1) ** (int(point[1] - y_line < 0)))
        else:
            trainingset_classification.append((-1) ** (int(point[0] > point1[0])))


    weights = numpy.array([0.0, 0.0, 0.0])

    #initially all points misclassified
    misclassified_points = range(0, numpoints)

    maxiters = 10000
    curriter = 0
    while len(misclassified_points) > 0 and curriter < maxiters:
        
        #pick index of misclassified point
        randindex = random.randrange(0, len(misclassified_points))
        
        #update w
        xcoord = trainingset[misclassified_points[randindex]][0]
        ycoord = trainingset[misclassified_points[randindex]][1]
        sign = trainingset_classification[misclassified_points[randindex]]
        weights = weights + sign * numpy.array([1.0, xcoord, ycoord])
        
        #update misclassified
        misclassified_points = []
        hyp_signs = []
        for index in range(0, numpoints):
            #compute h(x) = sign(w^T * x)
            curr_point = trainingset[index]
            curr_point_vec = numpy.array([1.0, curr_point[0], curr_point[1]])
            hypoth_sign = int((-1) ** (numpy.dot(weights, curr_point_vec) < 0))
            hyp_signs.append(hypoth_sign)
            
            #if misclassified, add this to list
            if hypoth_sign != trainingset_classification[index]:
                misclassified_points.append(index)
                
        curriter += 1

    return [curriter , target_slope , point1, weights]

########## end fcn pla ##########

def cross_validation(target_slope, target_point, weights):
    
    #generate random points in square
    numpoints = 1000
    disagree_count = 0
    lowbnd = -1
    upbnd = 1
    rng = upbnd - lowbnd
    
    for index in range(0, numpoints):
        point = (random.random() * rng + lowbnd, random.random() * rng + lowbnd)
        
        #classify point by target function
        if target_slope != numpy.Inf and target_slope != -numpy.Inf:
            y_val = target_slope * (point[0] - target_point[0]) + target_point[1]
            target_classification = (-1) ** (int(point[1] - y_val < 0))
        else:
            target_classification = (-1) ** (int(point[0] > target_point[0]))
        
        #classify point by hypothesis
        point_vec = numpy.array([1.0, point[0], point[1]])
        hypothesis_classification = (-1) ** (int(numpy.dot(weights, point_vec) < 0))
            
        if target_classification != hypothesis_classification:
            disagree_count += 1

    return 1.0 * disagree_count / numpoints
    
########## end function cross_validation ##########
    
num_runs = 1000
total_iters = 0
exp_prob_total = 0.0
for count in range(0,num_runs):
    ret_list = pla()
    total_iters += ret_list[0]
    exp_prob_total += cross_validation(ret_list[1], ret_list[2], ret_list[3])
    
print "total iters,", total_iters
print "avg iters", 1.0 * total_iters / num_runs
print "avg prob of disagreemtn between target and hyp:", exp_prob_total / num_runs



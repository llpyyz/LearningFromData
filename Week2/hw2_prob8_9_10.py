"""
Caltech Learning From Data
Homework 2 - Problems 8-10
Author: David Schonberger
Created: 10/12/2014
"""

#################
### Problem 8 ###
#################

def classify_nonlin_target(data):
    """
    Input:
    data, the set of training points
    
    Output:
    classifications, the set of classifications of the data
    
    Classifies according to the nonlinear fcn
    f(x1,x2) = sign(x1^2 + x2^2 - 0.6)
    """
    classifications = []
    for pt in data:
        sign = (-1) ** (int(pt[0] ** 2 + pt[1] ** 2 - 0.6 < 0))
        classifications.append(sign)
    
    return classifications
    
### Driver code for problem 8 #####

numsims = 1000
size = 1000
percent_to_flip = .1
num_to_flip = int(percent_to_flip * size)
misclassified_percent = 0.0

for sim in range(0, numsims):
    training_set = gen_training_set(size)
    ts_matrix = make_trainset_matrix(training_set)
    classifications = classify_nonlin_target(training_set)

    #introduce noise by flipping sign of random 10% of classifications
    permuted_indices = np.random.permutation(range(0, size))
    subset_indices = list(permuted_indices[0:num_to_flip])
    for idx in subset_indices:
        classifications[idx] *= -1
    
    #usual lin reg    
    weights = calc_weight_vec(ts_matrix, classifications)
    misclassified_percent += calc_E_in_2(ts_matrix, classifications, weights)    
    

print ""
print "#####"
print misclassified_percent
print "#####"
print ""


#    classifications = classify(training_set, m, pts[0])
#    weights = calc_weight_vec(ts_matrix, classifications)
#    iters += pla2(training_set, ts_matrix, classifications, weights)
#print training_set
#print ""
#print classifications
#print ""
#print permuted_indices
#print ""
#print subset_indices
#print ""
#print type(subset_indices)
#for idx in subset_indices:
#    print idx
#    print classifications[idx]
#    classifications[idx] *= -1
#    print classifications[idx]
#    
#print classifications


### End driver code for problem 8 #####



#######################################
##### begin problem 9 driver code #####
#######################################

numsims = 10
size = 1000
mcp_a = 0.0
mcp_b = 0.0
mcp_c = 0.0
mcp_d = 0.0
mcp_e = 0.0
mean_weights = array([0.0,0.0,0.0,0.0,0.0,0.0])
for sim in range(0, numsims):
    training_set = gen_training_set(size)
    training_set_nonlin = nonlin_transform(training_set)    
    ts_matrix = make_trainset_matrix(training_set_nonlin)
    classifications = classify_nonlin_target(training_set) #same target as in #8
    weights = calc_weight_vec(ts_matrix, classifications)
    mean_weights += weights
    
    #classify training set according to five possible hypotheses
    classify_a = classify_nonlin_a(training_set)
    mcp_a += calc_E_in_2(ts_matrix, classify_a, weights)
    
    classify_b = classify_nonlin_b(training_set)
    mcp_b += calc_E_in_2(ts_matrix, classify_b, weights)
    
    classify_c = classify_nonlin_c(training_set)
    mcp_c += calc_E_in_2(ts_matrix, classify_c, weights)
    
    classify_d = classify_nonlin_d(training_set)
    mcp_d += calc_E_in_2(ts_matrix, classify_d, weights)
    
    classify_e = classify_nonlin_e(training_set)
    mcp_e += calc_E_in_2(ts_matrix, classify_e, weights)
    
#print "hyp a res,", mcp_a / numsims
#print "hyp b res,", mcp_b / numsims
#print "hyp c res,", mcp_c / numsims
#print "hyp d res,", mcp_d / numsims
#print "hyp e res,", mcp_e / numsims
#print weights
print "ans to 9: ",mean_weights / numsims

#####################################
##### end problem 9 driver code #####
#####################################



##################
### Problem 10 ###
##################

########################################
##### begin problem 10 driver code #####
########################################


#estimate E_out using hyp found in #9 

weights = mean_weights / numsims #fixing w learned in #9!!!!

numsims = 1000
size = 1000
percent_to_flip = .1
num_to_flip = int(percent_to_flip * size)
E_out_est = 0.0

for sim in range(0, numsims):
    out_set = gen_training_set(size)
    
    out_set_nonlin = nonlin_transform(out_set)    

    out_matrix = make_trainset_matrix(out_set_nonlin)

    classifications = classify_nonlin_target(out_set) #same target as in #8

    #introduce noise by flipping sign of random % of classifications
    permuted_indices = np.random.permutation(range(0, size))
    subset_indices = list(permuted_indices[0:num_to_flip])
    for idx in subset_indices:
        classifications[idx] *= -1
    
    E_out_est += calc_E_in_2(out_matrix, classifications, weights)

print "out of sample error is approx", E_out_est / numsims

######################################
##### end problem 10 driver code #####
######################################




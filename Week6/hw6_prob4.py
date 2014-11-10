"""
Learning from Data
Problem 4
Adds weight decays to lin reg on non linear transformation from problem #2.
Uses lambda = 10 ^ (3).
Calculates E_in and E_out.
"""

import numpy as np

def do_nonlin_transform(data):
    """
    Input: 
    -data, a list of N points in training set
    
    Apply non linear trans:
    (x1,x2) -> (1, x2, x2, x1^2, x2^2, x1*x2, abs(x1 -x2), abs(x1 + x2))
    
    Output: list containing transformed points
    """
    nonlin_data = []
    for elt in data:
        l = [1, elt[0], elt[1], elt[0] ** 2, elt[1] ** 2, elt[0] * elt[1], abs(elt[0] - elt[1]), abs(elt[0] + elt[1])]
        nonlin_data.append(l)
        
    return nonlin_data

def make_matrix(data):
    """
    Input: 
    -data, a list of N points in training set
    
    Output:
    -ts_matrix, a N by d+1 matrix, where d = number
    of features in each point in data
    """
    nrow = len(data)
    ncol = len(data[0])
    ts_matrix = np.ones(nrow * ncol).reshape(nrow, ncol)
    for row_idx in range(0, nrow):
        for col_idx in range(1, ncol):
            ts_matrix[row_idx, col_idx] = data[row_idx][col_idx]
            
    return ts_matrix

#Uses regularization -> lambda required for weight decay
def regularization_calc_weight_vec(input_matrix, lam, output_vec):
    """
    Input: 
    -input_matrix, a N x (d+1) matrix where d is 
    the number of features for each input vector x. 
    (Each input vector x extra component x_0 = 1)
    
	-lam, the value of lambda related to the so-called 'soft-order' constraint
	on the solution
	
    -output_vec, a N by 1 vector of y values
    
    Output: w = inv(Z^T * Z + lam*I) * Z^T * output_vec
    """
    dim = input_matrix.shape[1]
    temp = np.dot(input_matrix.T, input_matrix) + lam * np.eye(dim)
    w = np.dot(np.dot(np.linalg.inv(temp), input_matrix.T), output_vec)
    return w
    

def calc_E(data_maxtrix, classifications, weights):
    """
    Input:
    -data_matrix, N x (d+1) array of points
    
    -classifications, the list of N correct classifications of each pt

    -weights, the weights found in lin reg, used to
    calculate sign(w^T * x) for each x in data_set
    
    Output: the fraction of points misclassified by the
    lin reg parameters. 
    """    
    hyp_vec = np.dot(data_maxtrix, weights)
    hyp_signs = np.array(map(int, hyp_vec >= 0))
    classify_signs = np.array(map(int, np.array(classifications) >= 0))
    
    return 1.0 * sum(hyp_signs != classify_signs)/len(hyp_signs)
    
#driver    
def __main__():
    #Data prep: read in text, convert to numbers
    #format of each line is (x1, x2, y) where y is +/-1
	#Assumes data files are in same dir as this file

    test_set = "./test_set.txt"
    train_set = "./training_set.txt"
    train_data = []
    test_data = []
    train_classifcations = []
    test_classifcations = []
    
    with open(train_set) as f:
        for line in f:
            train_data.append(line.split())
            
    
    with open(test_set) as f:
        for line in f:
            test_data.append(line.split())
        
    for elt in train_data:
        elt[0] = float(elt[0])
        elt[1] = float(elt[1])
        train_classifcations.append(int(float(elt.pop(2))))
    
    for elt in test_data:
        elt[0] = float(elt[0])
        elt[1] = float(elt[1])
        test_classifcations.append(int(float(elt.pop(2))))
    
    #get weights and calc E_in for training data
    #use lambda = 1000
    lam = 10 ** 3
    train_data_nonlin = do_nonlin_transform(train_data)
    train_matrix = make_matrix(train_data_nonlin)
    weights = regularization_calc_weight_vec(train_matrix, lam, train_classifcations)
    E_in = calc_E(train_matrix, train_classifcations, weights)
    print "E_in for training set", E_in
    
    #calc E_out for test data using weights found above
    test_data_nonlin = do_nonlin_transform(test_data)
    test_matrix = make_matrix(test_data_nonlin)
    E_out = calc_E(test_matrix, test_classifcations, weights)
    print "E_out for testw set", E_out
    
    
__main__()


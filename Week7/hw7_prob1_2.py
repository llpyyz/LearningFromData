"""
Learning from Data - Caltech/EdX
David Schonberger
HW 7
Validation
"""

"""
Learning from Data
HW 7, problems 1-5
Implements lin reg on non linear transformation
of training data set.
Splits input data set of 35 points into
--a set of 25 points and
--a set of 10 points.
Each sub set is used for training and for validation, in two different problems
Uses validation set to determine minimum 
classification error among several models
Then uses the out of sample test set to check out of sample error
and compare to validation error.
"""

import numpy as np

def do_nonlin_transform(data, num_features):
    """
    Input: 
    -data, a list of N points in training set
    -num_features, the number of features to use, from left to right
    
    Apply some or all of this non linear trans, depending on num_features:
    (x1,x2) -> (1, x1, x2, x1^2, x2^2, x1*x2, abs(x1 -x2), abs(x1 + x2))
    
    Output: list containing transformed points
    """
    nonlin_data = []
    for elt in data:
        l = [1, elt[0], elt[1], elt[0] ** 2, elt[1] ** 2, elt[0] * elt[1], abs(elt[0] - elt[1]), abs(elt[0] + elt[1])]
        temp = []
        for i in range(num_features + 1):
            temp.append(l[i])
            
        nonlin_data.append(temp)
        
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

def calc_weight_vec(input_matrix, output_vec):
    """
    Input: 
    -input_matrix, a N x (d+1) matrix where d is 
    the number of features for each input vector x. 
    (Each input vector x extra component x_0 = 1)
    
    -output_vec, a N by 1 vector of y values
    
    Output: w = pseudoinverse(input_matrix) * output_vec
    """
    w = np.dot(np.linalg.pinv(input_matrix), output_vec)
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

def read_data(fname, data_set):
    with open(fname) as f:
        for line in f:
            data_set.append(line.split())
    
    return data_set

def convert_data(data_set, classification_set):
    for elt in data_set:
        elt[0] = float(elt[0])
        elt[1] = float(elt[1])
        classification_set.append(int(float(elt.pop(2))))
        
    return [data_set, classification_set]

#split off num_val_pts from train_set into validation_set    
def split_train_set(train_set, num_val_pts, flag = "last"):
    """
    input:
    --train_set, the set of training data
    --num_val_pts, the number of points to peel off train_set
    --flag, a string indicating how to pick the pts for
    the validation set.
    'last' means pull pts from end of set
    'first' means pull from front of set
    output:
    --validation_set and reduced train_set
    """
    validation_set = []
    if flag == "last":
        train_set.reverse()
    
    for i in range(num_val_pts):
        validation_set.append(train_set.pop(0))
        
    return [train_set, validation_set]
    
#driver    
def __main__():
    #Data prep: read in text, convert to numbers
    #format of each line is (x1, x2, y) where y is +/-1
	#Assumes data files are in same dir as this file

    test_set = "./test_set.txt"
    train_set = "./training_set.txt"
    
    train_data = []
    val_data = []
    test_data = []
    
    train_classifications = []
    val_classifications = []
    test_classifications = []
    
    #### scenario, probs 1,2
#    num_val_pts = 10
#    val_data_loc = "last"
    
    #### scenario, probs 3,4
    num_val_pts = 25
    val_data_loc = "first"
    
    
    train_data = read_data(train_set , train_data)
    test_data = read_data(test_set , test_data)
    [train_data, train_classifications] = convert_data(train_data, train_classifications)
    
    [test_data, test_classifications] = convert_data(test_data, test_classifications)
    [train_data, val_data] = split_train_set(train_data, num_val_pts, val_data_loc)
    [train_classifications, val_classifications] = split_train_set(train_classifications, num_val_pts, val_data_loc)
    
       	
    #get weights and calc E_in for training data
    for num_features in range(3,8):
        #config training data, learn weights
        train_data_nonlin = do_nonlin_transform(train_data, num_features)
        train_matrix = make_matrix(train_data_nonlin)    
        weights = calc_weight_vec(train_matrix, train_classifications)
        
        #check erro on validation set
        val_data_nonlin = do_nonlin_transform(val_data, num_features)
        val_matrix = make_matrix(val_data_nonlin)
        E_val = calc_E(val_matrix, val_classifications, weights)
        print "\n\nE_val for val set and", num_features, "features is:", E_val
        
        #calc E_out for test data using weights found above
        test_data_nonlin = do_nonlin_transform(test_data, num_features)
        test_matrix = make_matrix(test_data_nonlin)
        E_out = calc_E(test_matrix, test_classifications, weights)
        print "E_out for test set", E_out, "\n\n"
   
   
__main__()


"""
############################################
Using 25 training pts and 10 validation pts:
############################################

E_val for val set and 3 features is: 0.3
E_out for test set 0.42 

E_val for val set and 4 features is: 0.5
E_out for test set 0.416 

E_val for val set and 5 features is: 0.2
E_out for test set 0.188 

E_val for val set and 6 features is: 0.0 <<< 6 features best on validation set but...
E_out for test set 0.084    

E_val for val set and 7 features is: 0.1
E_out for test set 0.072 <<< ...7 features produces smallest E_out

############################################
Using 10 training pts and 25 validation pts:
############################################

E_val for val set and 3 features is: 0.28
E_out for test set 0.396 

E_val for val set and 4 features is: 0.36
E_out for test set 0.388 

E_val for val set and 5 features is: 0.2
E_out for test set 0.284 

E_val for val set and 6 features is: 0.08 <<< smallest validation error for 6 features
E_out for test set 0.192 <<< smallst E_out for 6 features

E_val for val set and 7 features is: 0.12
E_out for test set 0.196 

Ans:
1) d
2) e
3) d
4) d
5) b
"""

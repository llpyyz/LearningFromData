"""
Learning From Data - EdX/Caltech
David Schonberger
Final Exam

Problem 10: Regularized lin reg for cassification.
Train 1-versus-5 classifier. 
Training data has two features.
Use lambda = 1 and 0.01, transformation on input data
Examine E_in, E_out, and overfitting
"""
import numpy as np

def read_data(fname, data_set):
    with open(fname) as f:
        for line in f:
            data_set.append(line.split())
    
    return data_set

def get_labels(data):
    labels = []
    for elt in data:
        labels.append(int(float(elt.pop(0))))
    return [labels, data]

def convert_str_data_to_float(data):
    for elt in data:
        for i in range(len(elt)):
            elt[i] = float(elt[i])

def one_v_one_labels(labels, data, label1, label2):
    one_v_one_labels = []
    reduced_data = []
    counter = 0
    for label in labels:
        if label == label1:
            one_v_one_labels.append(1)
            reduced_data.append(data[counter])
        elif label == label2:
            one_v_one_labels.append(-1)
            reduced_data.append(data[counter])
        counter += 1
    return [one_v_one_labels, reduced_data]


def do_nonlin_transform(data):
    """
    Input: 
    -data, a list of N points in training set
    
    Apply non linear trans:
    (x1,x2) -> (1, x2, x2, x1*x2, x1^2, x2^2)
    
    Output: list containing transformed points
    """
    nonlin_data = []
    for elt in data:
        l = [1, elt[0], elt[1], elt[0] * elt[1], elt[0] ** 2, elt[1] ** 2]
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

### driver code ###
#for 1-v-5 classifier
label1 = 1
label2 = 5

train_set = "C:/Users/David/Documents/TechStuff/OnlineCourses/Caltech_LearningFromData/Week9/features_train.txt"
train_data = []
train_labels = []
train_data = read_data(train_set , train_data)
[train_labels, train_data] = get_labels(train_data)
convert_str_data_to_float(train_data)
[train_labels, train_data] = one_v_one_labels(train_labels, train_data, label1, label2)

test_set = "C:/Users/David/Documents/TechStuff/OnlineCourses/Caltech_LearningFromData/Week9/features_test.txt"
test_data = []
test_labels = []
test_data = read_data(test_set , test_data)
[test_labels, test_data] = get_labels(test_data)
convert_str_data_to_float(test_data)
[test_labels, test_data] = one_v_one_labels(test_labels, test_data, label1, label2)

e_out_trans = []
e_in_trans = []

lambda_lst = [1, .01]
for lam in lambda_lst:
    print "\nlambda = ", lam
    ###With transform, E_in:
    train_data_nonlin = do_nonlin_transform(train_data)
    train_matrix = make_matrix(train_data_nonlin)
    weights = regularization_calc_weight_vec(train_matrix, lam, train_labels)
    curr_E_in = calc_E(train_matrix, train_labels, weights)
    e_in_trans.append(curr_E_in)    
    print "ein", curr_E_in
    
    ###With transform, E_out
    test_data_nonlin = do_nonlin_transform(test_data)
    test_matrix = make_matrix(test_data_nonlin)
    curr_E_out = calc_E(test_matrix, test_labels, weights)
    e_out_trans.append(curr_E_out)
    print "eout", curr_E_out
    
#print "e_in_trans", e_in_trans, "\n"
#print "e_out_trans", e_out_trans, "\n"

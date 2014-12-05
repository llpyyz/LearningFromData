"""
Learning From Data - EdX/Caltech
David Schonberger
Final Exam

Problem 7: Regularized lin reg for cassification.
Train one-versus-all classifiers for
digits 0-9. Training data has two features.
Use lambda = 1, no transformation on input data
Find lowest E_in across 5-v-all thru 9-v-all
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
        #labels.append(int(float(elt[0])))
        labels.append(int(float(elt.pop(0))))
    return [labels, data]

def convert_str_data_to_float(data):
    for elt in data:
        for i in range(len(elt)):
            elt[i] = float(elt[i])

def get_one_v_all_labels(labels, one_label):
    one_v_all_labels = []
    for label in train_labels:
        if label == one_label:
            one_v_all_labels.append(1)
        else:
            one_v_all_labels.append(-1)    
    return one_v_all_labels

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
train_set = "C:/Users/David/Documents/TechStuff/OnlineCourses/Caltech_LearningFromData/Week8/features_train.txt"
train_data = []
train_labels = []

num_of_labels = 10

train_data = read_data(train_set , train_data)
[train_labels, train_data] = get_labels(train_data)
convert_str_data_to_float(train_data)

e_in = []
lam = 1
for digit in range(num_of_labels):    
    one_v_all_labels = get_one_v_all_labels(train_labels , digit)
    train_matrix = make_matrix(train_data)
    weights = regularization_calc_weight_vec(train_matrix, lam, one_v_all_labels)
    curr_E_in = calc_E(train_matrix, one_v_all_labels, weights)
    print "E_in for ", digit," is:" , curr_E_in       

    e_in.append(curr_E_in)

    

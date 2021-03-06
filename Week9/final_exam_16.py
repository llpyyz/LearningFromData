"""
Learning From Data - EdX/Caltech
David Schonberger
Final Exam
Problem 16: 
Regualar RBF,
gamma = 1.5, K = 9 versus 12 clusters.
Target: f = sign(x_2 - x_1 +0.25*sin(pi*x_1))
input space: [-1,1] x [-1,1]
Generate 100 random pts uniformly for training.
Carry out sufficient runs to estimate
How E_in and E_out react as K goes from
9 to 12
Discount runs where empty clusters returned.
"""

from sklearn import svm, cluster
import numpy as np
import random
import math

def gen_pt(a = -1,b = 1,c = -1,d = 1):
    """
    Input: a, b, c, and d, the x and y bounds
    of a rectangle in the xy plane
    
    Output: a randomly generated point in [a,b] X [c,d]
    """
    return (random.random() * (b - a) + a, random.random() * (d - c) + c)

#labels using target 
#f = sign(x2 - x1 + 0.25*sin(pi*x1))
def gen_labels(data_set):
    """
    input:
    data_set, the data to be labeled
    
    output:
    labels, the set of +/-1 labels 
    """
    labels = []
    for pt in data_set:
        label = (-1) ** (int(pt[1] - pt[0] + 0.25 * math.sin(math.pi * pt[0]) < 0))
        labels.append(label)
        
    return labels

def do_svm(data, labels):
    clf = svm.SVC(C = cval, gamma = g, kernel = 'rbf')
    clf.fit(data, labels)
    labels_pred = clf.predict(data)
    return[clf, labels_pred]

def calc_reg_rbf_weights(centers, data, labels, gamma):
    """
    input:
    -centers, the K centers found by k-means
    -data, the data set to train the model on
    -labels, the labels for the data set
    -gamma, the rbf param
    
    output:
    -weights, an array of K + 1 weights for the hypothesis
    including a bias term
    """
    nrow = len(data)
    ncol = 1 + len(centers)
    mat = np.ones(nrow * ncol).reshape(nrow, ncol)
    for row_idx in range(0, nrow):
        for col_idx in range(1, ncol):
            vec = np.array(data[row_idx] - centers[col_idx - 1])
            mat[row_idx, col_idx] = math.exp(-gamma * np.dot(vec , vec))
            
    return np.dot(np.linalg.pinv(mat), labels)
    
def calc_pred_rbf_labels(centers, data, weights, gamma):
    """
    input:
    -centers, the k-means centers
    -data, the data set to be labeled
    -weights, the rbf weights
    -gamma, the rbf param
    
    output:
    -labels, the predicted labels for the data set
    """
    labels = []
    bias = weights.pop()
    for elt in data:
        val = bias        
        for i in range(len(weights)):
            vec = np.array(elt - centers[i])
            val += weights[i] * math.exp(-gamma * np.dot(vec, vec))
        sign = (-1) ** (int(val <= 0))
        labels.append(sign)    
    
    return labels

### begin driver code ###

g = 1.5
cval = 1000000
nosims = 100
empty_cluster_count = 0
N = 100
M = 1000
clust_lst = [9,12]
successful_runs = 0
e_in_diffs = []
e_out_diffs = []
result_tracker = [0,0,0,0,0]
for i in range(nosims):
    
    train_data = []
    for j in range(N):
        train_data.append(gen_pt())
    train_labels = gen_labels(train_data)
    
    #train regular RBF model:
    #use kmeans (lloyd's algo) to find centers then pseudoinv to calc weight
    e_in = []
    e_out = []
    for c in clust_lst:
        km = cluster.KMeans(n_clusters = c)
        km.fit(train_data)
        centers = km.cluster_centers_
        clust_labels = km.labels_
        empty_clusters = sum([num in clust_labels for num in range(c)]) < c
        if(empty_clusters):
            break
        
        #all clusters have pts ->
        #proceed to create model, cals e_in and e_out
        weights = list(calc_reg_rbf_weights(centers, train_data, train_labels, g))
        bias = weights.pop(0)        
        train_labels_pred_rbf = np.asarray(calc_pred_rbf_labels(centers, train_data, weights + [bias], g))    
        e_in.append(100.0 * sum(train_labels != train_labels_pred_rbf) / N)
    
        test_data = []
        for j in range(M):
            test_data.append(gen_pt())

        test_labels_actual = gen_labels(test_data)    
        test_labels_pred_rbf = np.asarray(calc_pred_rbf_labels(centers, test_data, weights + [bias], g))    
        e_out.append(100.0 * sum(test_labels_actual != test_labels_pred_rbf) / M)
        
    if(len(e_in) == len(clust_lst)):
        successful_runs += 1
        e_in_diffs.append(e_in[0] - e_in[1])
        e_out_diffs.append(e_out[0] - e_out[1])
        if(e_in[0] - e_in[1] > 0 and  e_out[0] - e_out[1] < 0):
            result_tracker[0] += 1
        elif(e_in[0] - e_in[1] < 0 and  e_out[0] - e_out[1] > 0):
            result_tracker[1] += 1
        elif(e_in[0] - e_in[1] < 0 and  e_out[0] - e_out[1] < 0):
            result_tracker[2] += 1
        elif(e_in[0] - e_in[1] > 0 and  e_out[0] - e_out[1] > 0):
            result_tracker[3] += 1
        elif(e_in[0] - e_in[1] == 0 and  e_out[0] - e_out[1] == 0):
            result_tracker[4] += 1
        
    
print "number of clusters:\n",clust_lst, "\n"    
print "e_in goes down",sum(np.asarray(e_in_diffs) > 0) 
print "e_in no change",sum(np.asarray(e_in_diffs) == 0)
print "e_in goes up",sum(np.asarray(e_in_diffs) < 0), "\n" 

print "e_out goes down",sum(np.asarray(e_out_diffs) > 0) 
print "e_out no change",sum(np.asarray(e_out_diffs) == 0)
print "e_out goes up",sum(np.asarray(e_out_diffs) < 0) 
print "\n\n",result_tracker

"""
Learning From Data - EdX/Caltech
David Schonberger
Final Exam
Problem 14/15: 
Hard margin SVM with
RBF kernel, versus regualar RBF,
gamma = 1.5, K = 9/12 clusters.
Target: f = sign(x_2 - x_1 +0.25*sin(pi*x_1))
input space: [-1,1] x [-1,1]
Generate 100 random pts uniformly for training.
Carry out sufficient runs to estimate
percentage of time SVM with RBF beats
regular RBF (Lloyd's + pinv). 
Discount runs where data set nonseparable
in Z space or where empty clusters returned.
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
nosims = 500
nonseparable = 0
empty_cluster_count = 0
N = 100
M = 1000
nclust = 9
#nclust = 12 for problem #15
svm_wins = 0
successful_runs = 0
for i in range(nosims):
    train_data = []
    for j in range(N):
        train_data.append(gen_pt())

    train_labels = gen_labels(train_data)
    
    #train SVM RBF model
    #return val = [svm model, predicted labels]
    svm_out = do_svm(train_data, train_labels)
    train_labels_pred = svm_out[1]
    
    #train regular RBF model:
    #use kmeans (lloyd's algo) to find centers then pseudoinv to calc weight
    km = cluster.KMeans(n_clusters = nclust)
    km.fit(train_data)
    centers = km.cluster_centers_
    clust_labels = km.labels_
    weights = list(calc_reg_rbf_weights(centers, train_data, train_labels, g))
    bias = weights.pop(0)
    
    nonsep_set = train_labels != list(train_labels_pred)
    empty_clusters = sum([num in clust_labels for num in range(nclust)]) < nclust
    if(nonsep_set):
        nonseparable += 1
    if(empty_clusters):
        empty_cluster_count += 1
    if(nonsep_set or empty_clusters):
        continue
    successful_runs += 1
    #all clusters have pts, training data separable under SVM model ->
    #proceed to evaluate test set and check which model has better E_out
    test_data = []
    for j in range(M):
        test_data.append(gen_pt())

    test_labels_actual = gen_labels(test_data)
    
    test_labels_pred_svm = svm_out[0].predict(test_data)
    e_out_svm = 100.0 * sum(test_labels_actual != test_labels_pred_svm) / M
    
    
    test_labels_pred_rbf = np.asarray(calc_pred_rbf_labels(centers, test_data, weights + [bias], g))
    e_out_rbf = 100.0 * sum(test_labels_actual != test_labels_pred_rbf) / M
    
    if(e_out_svm < e_out_rbf):
        svm_wins += 1
    

print "in", successful_runs, "good runs, svm has lower e_out than rbf",100.0 * svm_wins / successful_runs, "% of the time"

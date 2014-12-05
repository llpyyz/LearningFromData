"""
Learning From Data - EdX/Caltech
David Schonberger
Final Exam
Problem 13: 
Hard margin SVM with
RBF kernel, gamma = 1.5.
Target: f = sign(x_2 - x_1 +0.25*sin(pi*x_1))
input space: [-1,1] x [-1,1]
Generate 100 random pts uniformly for training.
Carry out sufficient runs to estimate
percentage of time data set not separable
in Z-space. Note: Separable implies E_in = 0
on training set.
"""

from sklearn import svm
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


### begin driver code ###

g = 1.5
cval = 1000000
nosims = 10000
nonseparable = 0
N = 100

for i in range(nosims):
    train_data = []
    for j in range(N):
        train_data.append(gen_pt())

    train_labels = gen_labels(train_data)

    clf = svm.SVC(C = cval, gamma = g, kernel = 'rbf')
    clf.fit(train_data, train_labels)
    train_labels_pred = clf.predict(train_data)
    if(train_labels != list(train_labels_pred)):
        nonseparable += 1

print "in", nosims, "runs:",100.0 * nonseparable / nosims, "% train data is non sep in Z"
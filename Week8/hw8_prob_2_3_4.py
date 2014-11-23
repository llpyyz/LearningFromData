"""
Learning From Data - EdX/Caltech
David Schonberger
HW #8
Porlbme 2-4: one-versus-all classifiers for
digits 0-9. Training data has two features.
The scikit-learn package is used to find
models, make predictions, and count SV's
for each of the ten possible classifiers
"""
from sklearn import svm
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
    
#test_set = "C:/Users/David/Documents/TechStuff/OnlineCourses/Caltech_LearningFromData/Week8/features_test.txt"
train_set = "C:/Users/David/Documents/TechStuff/OnlineCourses/Caltech_LearningFromData/Week8/features_train.txt"

train_data = []
train_labels = []
#test_data = []
#test_labels = []

num_of_labels = 10

train_data = read_data(train_set , train_data)
#test_data = read_data(test_set , test_data)

[train_labels, train_data] = get_labels(train_data)
#[test_labels, test_data] = get_labels(test_data)

convert_str_data_to_float(train_data)
#convert_str_data_to_float(test_data)

e_in = []
sv_count = []
for digit in range(num_of_labels):    
    one_v_all_labels = get_one_v_all_labels(train_labels , digit)
    clf = svm.SVC(C = 0.01, coef0 = 1.0, degree = 2, gamma = 1.0, kernel = 'poly')
    clf.fit(train_data, one_v_all_labels)
    sv_count.append(len(clf.support_vectors_))
    predicted_labels = clf.predict(train_data)
    e_in.append(1.0 * sum(predicted_labels != np.asarray(one_v_all_labels)) / len(predicted_labels))
    
print "in sample errors:", e_in
print "sv count", sv_count
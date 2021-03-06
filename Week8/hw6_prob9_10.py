"""
Learning From Data - EdX/Caltech
David Schonberger
HW #8
Problems 9-10: RBF kernel with 1-versus-5 classifier
for digits 0-9. Training data has two features.
Using RBF kernel with gamma = 1.
Checking E_in and E_out for C in {.01, 1, 100, 10^4, 10^6}
The scikit-learn package is used to find
models, make predictions.
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
        labels.append(int(float(elt.pop(0))))
    return [labels, data]

def convert_str_data_to_float(data):
    for elt in data:
        for i in range(len(elt)):
            elt[i] = float(elt[i])

def get_one_v_all_labels(labels, one_label):
    one_v_all_labels = []
    for label in labels:
        if label == one_label:
            one_v_all_labels.append(1)
        else:
            one_v_all_labels.append(-1)    
    return one_v_all_labels

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

### begin driver code ###

test_set = "C:/Users/David/Documents/TechStuff/OnlineCourses/Caltech_LearningFromData/Week8/features_test.txt"
train_set = "C:/Users/David/Documents/TechStuff/OnlineCourses/Caltech_LearningFromData/Week8/features_train.txt"

train_data = []
train_labels = []
test_data = []
test_labels = []


#for 1-v-5 classifier
label1 = 1
label2 = 5

train_data = read_data(train_set , train_data)
test_data = read_data(test_set , test_data)

[train_labels, train_data] = get_labels(train_data)
[test_labels, test_data] = get_labels(test_data)

convert_str_data_to_float(train_data)
convert_str_data_to_float(test_data)

[one_versus_one_train_labels, one_versus_one_train_data] = one_v_one_labels(train_labels, train_data, label1, label2)
[one_versus_one_test_labels, one_versus_one_test_data] = one_v_one_labels(test_labels, test_data, label1, label2)

    
C_list = [0.01, 1, 100, 10000, 1000000]

e_in = []
e_out = []
sv_count = []

for C_val in C_list:
    clf = svm.SVC(C = C_val, gamma = 1.0, kernel = 'rbf')        
    clf.fit(one_versus_one_train_data, one_versus_one_train_labels)
    sv_count.append(len(clf.support_vectors_))
    
    predicted_train_labels = clf.predict(one_versus_one_train_data)
    e_in.append(1.0 * sum(predicted_train_labels != np.asarray(one_versus_one_train_labels)) / len(predicted_train_labels))
    
    predicted_test_labels = clf.predict(one_versus_one_test_data)
    e_out.append(1.0 * sum(predicted_test_labels != np.asarray(one_versus_one_test_labels)) / len(predicted_test_labels))
            
print "in sample errors:"
for i in range(len(C_list)):
    print e_in[i]
print "\n\n"

print "out of sample errors:"
for i in range(len(C_list)):
    print e_out[i]
print "\n\n"

print "sv count:"
for i in range(len(C_list)):
    print sv_count[i]
print "\n\n"

"""
Learning From Data - EdX/Caltech
David Schonberger
HW #8
Problems 7-8: Cross validation of 1-versus-5 classifier
for digits 0-9. Training data has two features.
Using quadratic polynomial kernel.
Use E_cv to pick best value of C in {.0001, .001, .01, .1, 1}.
The scikit-learn package is used to find
models, make predictions.
"""
from sklearn import svm
import numpy as np
import random as rnd
import copy

def read_data(fname, data_set):
    with open(fname) as f:
        for line in f:
            data_set.append(line.split())
    
    return data_set

#returns labels split out from data
#labels are numeric, but returned data
#is still str
def get_labels(data):
    labels = []
    for elt in data:
        labels.append(int(float(elt.pop(0))))
    return [labels, data]

def convert_str_data_to_float(data):
    for elt in data:
        for i in range(len(elt)):
            elt[i] = float(elt[i])
    return data

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

#breaks up data list into roughly equal size sublists
def make_n_sublists(data, num_lists):
    partitioned_data = []
    #data_copy = list(data)
    data_copy = copy.deepcopy(data)
    sublist_size = len(data) / num_lists
    extra_pts = len(data) % num_lists
    #peel off pts to each sublist
    for count in range(num_lists):
        temp_lst = []
        for count2 in range(sublist_size):
            temp_lst.append(data_copy.pop(0))
        partitioned_data.append(temp_lst)
        
    #add any extra pts
    for count in range(extra_pts):
        partitioned_data[count].append(data_copy.pop(0))
        
    return partitioned_data

#Pulls out a given sublist and callers helper to combine
#remaining sublists into one big list of data
def extract_sublist(data, elt_number):
    data_copy = list(data)
    sublist = data_copy.pop(elt_number) 
    return [sublist, combine_sublists(data_copy)]

#Helper fcn, called by extract_sublist()
#Takes a list of sublists and combines 
#them into a single list
def combine_sublists(data):
    flat_list = []
    for lst in data:
        for elt in lst:
            flat_list.append(elt)
    return flat_list

def one_v_one_filter(data, label1, label2):
    filtered_data = []
    for elt in data:
        val = int(float(elt[0]))
        if val == label1 or val == label2:
            filtered_data.append(elt)
    
    return filtered_data

#########################
### begin driver code ###
#########################

test_set = "C:/Users/David/Documents/TechStuff/OnlineCourses/Caltech_LearningFromData/Week8/features_test.txt"
train_set = "C:/Users/David/Documents/TechStuff/OnlineCourses/Caltech_LearningFromData/Week8/features_train.txt"

train_data = []
train_labels = []
test_data = []
test_labels = []

num_runs = 100
cv_fold = 10
deg = 2

#for 1-v-5 classifier
label1 = 1
label2 = 5

C_list = [0.0001, 0.001, 0.01, 0.1, 1]

lowest_cv = []
avg_cv = []
for i in range(len(C_list)):
    lowest_cv.append(0)
    avg_cv.append(0.0)


train_data = read_data(train_set , train_data)
test_data = read_data(test_set , test_data)

train_data_filtered = one_v_one_filter(list(train_data), label1, label2)
#print "\n***before runs begin, filtered data:", train_data_filtered[0:5], "\n"

for run in range(num_runs):

    train_data_partitioned_numeric = []
    train_labels_partitioned = []

    train_filtered_copy = list(train_data_filtered)
        
    train_filtered_copy = rnd.sample(train_filtered_copy, len(train_filtered_copy))
    
    train_data_partitioned_str = make_n_sublists(train_filtered_copy, cv_fold)
    
    #split labels from features and convert to numeric
    for elt in train_data_partitioned_str:
        res = get_labels(elt)
        train_labels_partitioned.append(res[0])
        train_data_partitioned_numeric.append(convert_str_data_to_float(res[1]))
    
    #iterate over all values of C
    E_cv_lst = []
    for C_val in C_list:
        #do n-fold CV
        E_cv_curr = 0.0 #cumulative cv error for this value of C
        for curr_cv_elt in range(cv_fold):
            [cv_data, rest_of_data] = extract_sublist(train_data_partitioned_numeric , curr_cv_elt)
            
            [cv_labels, rest_of_labels] = extract_sublist(train_labels_partitioned , curr_cv_elt)
            
            clf = svm.SVC(C = C_val, coef0 = 1.0, degree = deg, gamma = 1.0, kernel = 'poly')
            
            clf.fit(rest_of_data, rest_of_labels)
            
            predicted_cv_labels = clf.predict(cv_data)
            
            E_cv_curr += 1.0 * sum(predicted_cv_labels != np.asarray(cv_labels)) / len(cv_labels)
            
        E_cv_lst.append(E_cv_curr / cv_fold) #append E_cv for curr ent val of C
            
    min_cv = min(E_cv_lst)

    idx = E_cv_lst.index(min_cv)

    lowest_cv[idx] += 1 #chalk one up for the winner in this run

    avg_cv = [x + y for x, y in zip(avg_cv, E_cv_lst)]

print "***\n\ncounts on which C won each run:" , lowest_cv
print "mean E_cv over", num_runs, "runs:", [elt * 1.0 / num_runs for elt in avg_cv]

"""
Learning From Data - EdX/Caltech
David Schonberger
Final Exam
Problem 12: Same fixed data set as in
#11. This time, use hard margin SVM
with a second order polynomial kernel
How any SV's?
"""
from sklearn import svm

data = [[1,0], [0,1], [0,-1], [-1,0], [0,2], [0,-2], [-2,0]]
labels = [-1,-1,-1,1,1,1,1]
c_val = 1000000
clf = svm.SVC(C = c_val, gamma = 1.0, degree = 2, kernel = 'poly')
clf.fit(data, labels)
print len(clf.support_vectors_), "SVs:"
print clf.support_vectors_
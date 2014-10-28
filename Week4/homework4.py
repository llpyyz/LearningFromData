"""
EdX Learning from Data (Caltech)
Author : David Schonberger
Created: 10/2/2014
"""

"""
HW 4 , Problem #1
d_VC = 10. Want 95% confidence that generalization error
is <= 0.05. Find smallest N for which this is true.
Approx m_H(N) by N^d_VC
"""
import math
N = 1
res = 10 * math.log(N) - .0025 * N / 8 + math.log(80 * 2 ** 10)
while res > 0 and N <= 500000:
    N += 1
    res = 10 * math.log(N) - .0025 * N / 8 + math.log(80 * 2 ** 10)
    
    
print "N = ", N, "res = ", res #about N = 452957



"""
EdX Learning from Data (Caltech)
HW 4 , Problem #2
d_VC = 50; delta = 0.05. 
Plot several bounds on epsion,
the generalization error, as fcn of N
where N is large, say N = 10000
"""


import numpy as np
import matplotlib.pyplot as plt

d_vc = 50
delta = 0.05
x = np.linspace(2.0 * d_vc, 10000.0, num = 100)

######################
#bound 1 - original VC 
######################
y1 = np.sqrt(8.0 / x * (np.log(4 / delta * 2 ** d_vc) + d_vc * np.log(x)))
plt.plot(x,y1)
plt.show()

print y1[90:99] #last val == .635

##############################
#bound #2 - Rademacher penalty
##############################
y2 = np.sqrt((2 * np.log(2 * x) + 2 * d_vc * np.log(x)) / x) + np.sqrt(2 / x * np.log(1 / delta)) + 1.0 / x
plt.plot(x, y2)
plt.show()
print y2[90:99] #last val == 0.333

########################################################################
#bound 3 - Parrondo and Van den Broek 
#(implicit bnd; solved for epsilon to get explicit quadratic ineqaulity)
########################################################################

y3 = (1 + np.sqrt(1 + x * (np.log(6 / delta) + d_vc * np.log(2 * x)))) / x
plt.plot(x, y3)
plt.show()
print y3[90:99] # last val == 0.225

##############################################
#bound 4 - Devroye
#(also implicit; solved for epsilon as before)
##############################################

rad = 4 + (2 * x - 4) * (np.log(4 / delta) + 2 * d_vc * np.log(x))
y4 = (2 + np.sqrt(rad))/ (2 * x - 4)
plt.plot(x, y4)
plt.show()
print y4[90:99] # last val == 0.216


"""
EdX Learning from Data (Caltech)
HW 4 , Problem #3
d_VC = 50; delta = 0.05. 
Plot several bounds on epsion,
the generalization error, as fcn of N
where N is small, say N = 5 (< d_vc)
"""

######################
#bound 1 - original VC 
######################

def calc_vc_bnd(N, d):
    return np.sqrt(8.0 / N * np.log(4 * 2 ** (2 * N) / d))
    
##############################
#bound #2 - Rademacher penalty
##############################

def calc_rad_bnd(N, d):
    return np.sqrt(2.0 / N * np.log(2 * N * 2 ** N)) + np.sqrt(2.0 / N * np.log(1 / d)) + 1.0 / N

########################################################################
#bound 3 - Parrondo and Van den Broek 
#(implicit bnd; solved for epsilon to get explicit quadratic ineqaulity)
########################################################################

def calc_parr_vbd_bnd(N, d):
    return (1 + np.sqrt(1 + N * np.log(6 * 2 ** (2 * N)/ d)))/ N

##############################################
#bound 4 - Devroye
#(also implicit; solved for epsilon as before)
##############################################

def calc_dev_bnd(N, d):
    return (2 + np.sqrt(4 + (2 * N - 4) * np.log(4 / d * 2 ** (N * N))))/ (2 * N - 4)

minN = 3
maxN = 20

for N in range(minN, maxN):
    print "For N = ", N
    print calc_vc_bnd(N, delta)
    print calc_rad_bnd(N, delta)
    print calc_parr_vbd_bnd(N, delta)
    print calc_dev_bnd(N, delta)
    print "*****"
    print
    
"""
EdX Learning from Data (Caltech)
HW 4 , Problem #4
"""

#a = np.linspace(0,1,num = 500)
#y = -2 / np.pi * np.cos(np.pi * a) + (1 - 2 * a ** 2) * np.sin(np.pi * a)/ (2 * a)
#plt.plot(a, y)
#plt.show()
#
#yprime = 2 * np.sin(np.pi * a) - (np.sin(np.pi * a) + a * np.pi * np.cos(np.pi * a)) + (2 * a * np.pi * np.cos(np.pi * a) - 2 * np.sin(np.pi * a))/(4 * a ** 2)
#plt.plot(a, yprime)
#plt.show()
#
#import math
#from scipy.optimize import fsolve
#def func(a):
#    return 2 * np.sin(np.pi * a) - (np.sin(np.pi * a) + a * np.pi * np.cos(np.pi * a)) + (2 * a * np.pi * np.cos(np.pi * a) - 2 * np.sin(np.pi * a))/(4 * a ** 2)
#    
#root = fsolve(func, 0.7)
#print root
#

#print .5*(1 - 6 / np.pi ** 2)
#print np.pi/3 * np.sin(3)
#print 3 / np.pi

N = 10
M = 10
low = -1
high = 1
l = []

#Method 1 :
#lin reg to fit line thru origin to two random points
# on f(x) = sin(pi * x) over domain [-1, 1]

print "***begin method 1***"
for idx1 in range(M):
    
    accum_slope = 0.0    
    for idx2 in range(N):
        
        #gen two points on f(x)
        x1 = random.random() * (high - low) + low
        y1 = np.sin(np.pi * x1)
        x2 = random.random() * (high - low) + low
        y2 = np.sin(np.pi * x2)
    
        #make matrix
        X = np.array([[x1] , [x2]])
        y = np.array([[y1], [y2]])
        
        #calc weights
        w = np.dot(np.linalg.pinv(X), y)
        
        #update slope accumulator
        accum_slope += w[0][0]

    l.append(accum_slope * 1.0 / N)    
    
m_bar_est  = sum(l) * 1.0 / len(l)
print "for", M, "runs of", N , "trials, the mean slope is", m_bar_est

sq_dev = 0.0
for idx in range(len(l)):
    sq_dev += (l[idx] - m_bar_est) ** 2

sq_dev = np.sqrt(sq_dev/(M * (M-1))) 

print "ci for est mean slope", m_bar_est - 5 * sq_dev, " ", m_bar_est + 5 * sq_dev

print "***end method 1***"
print

#Method 2:
#Pick two points, for each find a line thru origin
#Whichever line yields smallest mean sq error, use that
#as hypothesis. Then repeat a number of times to get
#avg hypothesis gbar
N = 10
M = 10
l = []
print "***begin method 2***"
for idx1 in range(M):
    
    accum_slope = 0.0
    for idx2 in range(N):
        
        #gen two points on f(x), calc error of each
        x1 = random.random()
        y1 = np.sin(np.pi * x1)
        a1 = y1 / x1
        err1 = a1 ** 2 / 3 - 2 * a1 / np.pi + .5
        
        x2 = random.random()
        y2 = np.sin(np.pi * x2)
        a2 = y2 / x2
        err2 = a2 ** 2 / 3 - 2 * a2 / np.pi + .5
        
        if err1 < err2:
            accum_slope += a1
        else:
            accum_slope += a2
            
    l.append(accum_slope * 1.0 / N)

        
m_bar_est  = sum(l) * 1.0 / len(l)
print "for", M, "runs of", N , "trials, the mean slope is", m_bar_est

sq_dev = 0.0
for idx in range(len(l)):
    sq_dev += (l[idx] - m_bar_est) ** 2

sq_dev = np.sqrt(sq_dev/(M * (M-1))) 

print "ci for est mean slope", m_bar_est - 5 * sq_dev, " ", m_bar_est + 5 * sq_dev
    
print "***end method 2***"
print


"""
EdX Learning from Data (Caltech)
HW 4 , Problem #5
Estimate bias
"""
#using slope from #4:
#err formula (found via integration, p(m) = 1/3 * m^2 - 2/pi * m + .5
m = 1.42819585038
mse = m**2 / 3 - 2 * m / np.pi + .5
print mse

"""
EdX Learning from Data (Caltech)
HW 4 , Problem #6
Estimation of variance from #4
"""

#warm up: verification of variance for H_0 in slide
N = 10000
low = -1
high = 1
accum = 0.0
for idx in range(N):
    
    x1 = random.random() * (high - low) + low
    y1 = np.sin(np.pi * x1)
    x2 = random.random() * (high - low) + low
    y2 = np.sin(np.pi * x2)

    yavg = (y1 + y2) / 2.0
    accum += yavg ** 2
    
accum /= N
print 
print "var", accum
print

#Problem 6:

N = 10000
accum_slope = 0.0
l1 = []
for idx in range(N):
        
    #gen two points on f(x)
    x1 = random.random() * (high - low) + low
    y1 = np.sin(np.pi * x1)
    x2 = random.random() * (high - low) + low
    y2 = np.sin(np.pi * x2)

    #make matrix
    X = np.array([[x1] , [x2]])
    y = np.array([[y1], [y2]])
    
    #calc weights
    w = np.dot(np.linalg.pinv(X), y)
    l1.append(w[0][0])
    
    #update slope accumulator
    accum_slope += w[0][0]

est_slope  = accum_slope / N
print "est of gbar",est_slope #about 1.42

est_var = 0.0
for elt in l1:
    est_var += (elt - est_slope) ** 2
    
est_var /= (3 * N)
print "est var", est_var #about .24


"""
EdX Learning from Data (Caltech)
HW 4 , Problem #7
Min E_out for several H
"""

############
#a) h(x) = b 
############
#from slides, E_out = 0.5 + 0.25 = 0.75

############
#b) h(x)= ax 
############
m = 1.42819585038
bias = m**2 / 3 - 2 * m/np.pi + .5
print "bias for h(x) = ax", bias #***about 0.27***
print
#from above, E_out  = .24 + .27 = 0.51

#################
#c) h(x) = ax + b
#################
#from slides, E_out = .21 + 1.69 = 1.9

################
#d) h(x) = ax^2
################
#Lin reg to fit g(x) = ax^2 to two random points
# on f(x) = sin(pi * x) over domain [-1, 1]

from scipy import integrate
N = 5
M = 5
low = -1
high = 1
l = []
l2 = []
for idx2 in range(M):    
    accum_coeff = 0.0
    for idx1 in range(N):
        
        #gen two points on f(x)
        x1 = random.random() * (high - low) + low
        y1 = np.sin(np.pi * x1)
        x2 = random.random() * (high - low) + low
        y2 = np.sin(np.pi * x2)
    
        #make matrix
        X = np.array([[x1 ** 2] , [x2 ** 2]])
        y = np.array([[y1], [y2]])
        
        #calc weights
        w = np.dot(np.linalg.pinv(X), y)
        l2.append(w[0][0])

        #update accumulator
        accum_coeff += w[0][0]
    
    accum_coeff /= N
    l.append(accum_coeff)

final_coeff = sum(l) / len(l)
print
print "for", M , "runs of", N , "trials each, avg coeff for g(x) = ax^2:" , final_coeff
print

mse_fcn = lambda x: (np.sin(np.pi * x) - final_coeff * x ** 2 ) ** 2
bias = integrate.quad(mse_fcn, -1, 1)[0]  / 2.0
print
print "for g(x) = ax^2, bias = ", bias #***about 0.5***
print

est_var = 0.0
for elt in l2:
    est_var += (elt - final_coeff) ** 2

est_var  = est_var * 2 / ( 5 * M * N)
print
print "est var", est_var #***about 30-40***
print


####################
#d) h(x) = ax^2 + b
####################
#Lin reg to fit g(x) = ax^2 + b to two random points
# on f(x) = sin(pi * x) over domain [-1, 1]

N = 25000
M = 1
low = -1
high = 1
l = []
l2 = []
for idx2 in range(M):    
    accum_coeff = [0.0, 0.0]
    for idx1 in range(N):
        
        #gen two points on f(x)
        x1 = random.random() * (high - low) + low
        y1 = np.sin(np.pi * x1)
        x2 = random.random() * (high - low) + low
        y2 = np.sin(np.pi * x2)
    
        #make matrix
        X = np.array([[1, x1 ** 2] , [1, x2 ** 2]])
        y = np.array([[y1], [y2]])
        
        #calc weights
        w = np.dot(np.linalg.pinv(X), y)
        l2.append((w[0][0], w[1][0]))

        #update accumulator
        accum_coeff[0] += w[0][0] #b, constant coeff
        accum_coeff[1] += w[1][0] #a, quadratic coeff
    
    accum_coeff[0] /= N
    accum_coeff[1] /= N
    
    l.append((accum_coeff[0], accum_coeff[1]))
    
sum1 = 0.0
sum2 = 0.0
for elt in l:
    sum1 += elt[0]
    sum2 += elt[1]
    
final_coeffs = (sum1 / len(l), sum2 / len(l))
print
print "coeffs for g(x) = ax^2 + b:", final_coeffs
print

mse_fcn = lambda x: (np.sin(np.pi * x) - (final_coeffs[1] * x ** 2 + final_coeffs[0]) ) ** 2
bias = integrate.quad(mse_fcn, -1, 1)[0]  / 2.0
print
print "for g(x) = ax^2 + b, bias = ", bias #***about ***
print


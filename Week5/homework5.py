"""
EdX Learning from Data (Caltech)
HW 5 
Author : David Schonberger
Created: 11/2/2014
"""

import math

### problem 5 ###
# minimize given error surface via gradient descent

def errfcn(u,v):
    return (u * math.exp(v) - 2 * v * math.exp(-u)) ** 2
    
#init
u = 1
v = 1

#learning rate for GD
eta = 0.1

maxiters = 30
count = 0
errval = errfcn(u,v)
tol = 10 ** (-14)
while count < maxiters:
    delta_u = -eta * 2 * (u * math.exp(v) - 2 * v * math.exp(-u)) * (math.exp(v) + 2 * v * math.exp(-u))
    delta_v = -eta * 2 * (u * math.exp(v) - 2 * v * math.exp(-u)) * (u * math.exp(v) - 2 * math.exp(-u))
    u += delta_u
    v += delta_v
    errval = errfcn(u,v)
    count += 1
    
print count , "iters; err", errval
print "u:", u, "v:", v


### problem 6 ###
# minimize given error surface via 'coordinate' descent,
#moving first in u direction, then in v direction

#init
u = 1
v = 1

#learning rate for GD
eta = 0.1

maxiters = 30
count = 0
errval = errfcn(u,v)

while count < maxiters:
    delta_u = -eta * 2 * (u * math.exp(v) - 2 * v * math.exp(-u)) * (math.exp(v) + 2 * v * math.exp(-u))
    u += delta_u
    
    delta_v = -eta * 2 * (u * math.exp(v) - 2 * v * math.exp(-u)) * (u * math.exp(v) - 2 * math.exp(-u))    
    v += delta_v
    
    errval = errfcn(u,v)
    count += 1
    
print count , "iters; err", errval
print "u:", u, "v:", v


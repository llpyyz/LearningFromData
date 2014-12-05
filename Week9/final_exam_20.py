"""
Learning From Data - EdX/Caltech
David Schonberger
Final Exam
Problem 20: 
Give two learned hypotheses g1 and g2, 
construct g(x) = .5(g1(x) + g2(x)), for all x in X
Using mse to measure E_out, this simluation explores:
1) whether E_out(g) is worse than avg of E_out(g1) and E_out(g2)
2) whether E_out(g) has to be between E_out(g1) and E_out(g2).
This is a simulation for a specific target, f = x^2 for x in [0,1]. 
The data set D is 2 random uniform points on the target. The two
hypothesies are:
g1(x) = b, a constant
g2(x) = ax+b, a line
g1 is calculated as the avrage of the y-coords of pts in D
g2 is calculated as the line passing through pts in D.
The E_out (mse) formula was calculated using calculus
and here the needed values are merely plugged in.
The simulation counts how many times E_out(g)
is <= the avg of the E_outs and how many times
it falls between them.
"""

import random

nosims = 10000
low = 0
high = 1
lte_avg_e_out = 0
betw_two_e_outs = 0
for i in range(nosims):
    
    x1 = random.random() *(high - low) + low
    x2 = random.random() *(high - low) + low
    p1 = [x1, x1**2]
    p2 = [x2, x2**2]
    b = (p1[1] + p2[1])/2
    e_out_g1 = .2 - 2*b/3 + b **2
    
    
    slope = (p1[1] - p2[1])/(p1[0] - p2[0])
    intercept = p1[1] - slope * p1[0]
    e_out_g2 = 0.2 - 0.5*slope + (slope ** 2 - 2*intercept)/3 + slope * intercept + intercept ** 2
    
    avg_e_out = (e_out_g1 + e_out_g2) / 2    
    
    slope2 = .5 * slope
    intercept2 = (b + intercept) / 2
    e_out_avg = 0.2 - 0.5*slope2 + (slope2 ** 2 - 2*intercept2)/3 + slope2 * intercept2 + intercept2 ** 2
    if(e_out_avg <= avg_e_out):
        lte_avg_e_out += 1
        
    if(e_out_avg >= min(e_out_g1, e_out_g2) and e_out_avg <= max(e_out_g1, e_out_g2)):
        betw_two_e_outs += 1

print "***\nin", nosims, "runs:\n","lte_avg_e_out:\n",lte_avg_e_out, "\nbetw_two_e_outs:\n",betw_two_e_outs
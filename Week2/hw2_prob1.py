import math
import random
from operator import mul
from numpy import *
import timeit


"""
Caltech - Learning from Data
HW #2
Author: David Schonberger
Created: 10/9/2014
"""

"""
Problem 1 
"""
def run_trials(num_trials, num_coins, flips_per_coin):
    """
    Input: 
    -num_trials, the number of overall trials to simulate
    -num_coins, the number of coins to flip each trial
    -flips_per_coin, the number of times to flip each
    coin each trial.
    
    Purpose:
    Run 'num_trials' trials, where in each trial
    'num_coins' coins are each flipped 'flips_per_coin'
    times and the % of heads is recorded.
    
    Returns a dictionary with one key for each coin
    and a list of 'num_trials' heads freqs as the
    value for each coin.
    """
    data_dict = {}
    data_dict['first'] = []
    data_dict['rand'] = []
    data_dict['min'] = []

    for trial in range(0, num_trials):
        rnd = random.randrange(0, num_coins)
        minfreq = 1.1
        firstfreq = 0.0
        randfreq = 0.0
        for coin in range(0, num_coins):
            head_freq = 0
            for flip in range(0, flips_per_coin):
                if random.random() <= 0.5:
                    head_freq += 1
                    
            if coin == 0:
                data_dict['first'].append(head_freq * 1.0 / flips_per_coin) #first coin
            elif coin == rnd:
                data_dict['rand'].append(head_freq * 1.0 / flips_per_coin) #rand coin
            
            minfreq = min(minfreq, head_freq * 1.0 / flips_per_coin)
            
        data_dict['min'].append(minfreq)
                
    return data_dict        

def avg_of_min_coin(data_dict):
    """
    """
    avg = 0.0
    num_trials = len(data_dict.values()[0])
    for trial in range(0, num_trials):
        avg += min([x[trial] for x in data_dict.values()])
    return avg / num_trials
    
nt = 10000
nc = 1000
nf = 10

d = run_trials(nt, nc, nf)
#firstelts = [x[0] for x in d.values()]
#firstmin = min(firstelts)
#print d
nu_min = avg_of_min_coin(d)
print nu_min

print d['rand'][0:100]
print sum(d['rand'])/nt
print sum(d['first'])/nt

print ""
print "#####"
eps = .2
rand_l1 = [abs(x - 0.5) for x in d['rand']]
rand_l2 = filter(lambda x : x > eps, rand_l1)
rand_prob1 =  len(rand_l2) * 1.0 / nt
print rand_prob1
rand_hoeff1 = 2 * math.exp(-2 * (eps ** 2) * nf)
print rand_hoeff1


first_l1 = [abs(x - 0.5) for x in d['first']]
first_l2 = filter(lambda x : x > eps, first_l1)
first_prob1 =  len(first_l2) * 1.0 / nt
print first_prob1
first_hoeff1 = 2 * math.exp(-2 * (eps ** 2) * nf)
print first_hoeff1

min_l1 = [abs(x - 0.5) for x in d['min']]
min_l2 = filter(lambda x : x > eps, min_l1)
min_prob1 =  len(min_l2) * 1.0 / nt
print min_prob1
min_hoeff1 = 2 * math.exp(-2 * (eps ** 2) * nf)
print min_hoeff1

print "#####"
print ""

#########
#prob 3-4
#########
mu = .1 #prob h misses target f
lam = .5 #prob 
print (1 - lam) * mu + lam * (1 - mu)
print (1 - lam)*(1 - mu) + lam * mu

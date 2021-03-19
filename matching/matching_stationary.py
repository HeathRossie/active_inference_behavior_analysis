# using EFE action selection
# stationary version

import numpy as np
from scipy.special import digamma, betaln
from scipy.special import betaincinv
import matplotlib.pyplot as plt
import pandas as pd
import math

# calcurate expected free energy
# code adapted from https://github.com/dimarkov/aibandits
def G(alpha_t, beta_t, alpha):
    nu_t = alpha_t + beta_t
    mu_t = alpha_t / nu_t
    
    KL_a = - betaln(alpha_t, beta_t) + (alpha_t - alpha) * digamma(alpha_t)\
             + (beta_t - 1) * digamma(beta_t) + (alpha + 1 - nu_t) * digamma(nu_t)
    
    H_a = - mu_t * digamma(alpha_t + 1) - (1-mu_t) * digamma(beta_t + 1) + digamma(nu_t + 1)
    
    return KL_a + H_a #+ np.random.normal(0, .1, 1)


# action selection based on expected free energy
def choice(alpha_t, beta_t, alpha):
    G_a = G(alpha_t, beta_t, alpha)
    prob = softmax(-G_a)
    choice = np.random.multinomial(1, prob, size=1)
    c = np.where(choice[0] == 1)
    return int(list(c)[0])


# softmax function
def softmax(a):
    a_max = max(a)
    x = np.exp(a-a_max)
    u = np.sum(x)
    return x/u


# matching experiment
def sim_matching(VI1, VI2):
        
    ### simulation ###
    # set config
    NUM_RFT = 1000
    MINIMUM = 1
    #VI1 = 30
    #VI2 = 120
    DT = 1.
    lam = 1.5
    alpha = math.exp(2 * lam)
    
    # set RFT time (two VI schedules)
    RFT_TIME1 = np.random.exponential(VI1 - MINIMUM, NUM_RFT) + MINIMUM
    RFT_TIME2 = np.random.exponential(VI2 - MINIMUM, NUM_RFT) + MINIMUM

    # initialize variables
    resp1 = np.array([])
    resp2 = np.array([])
    alpha_t = np.array([1., 1.])
    beta_t = np.array([1., 1.])
    timer1 = 0
    timer2 = 0
    rft = 0 
    rft_current = [0, 0]
    alpha = math.exp(2 * lam)
    G_seq = G(alpha_t, beta_t, alpha)
    timer_global = 0

    while True:
        # update time
        timer1 = timer1 + DT
        timer2 = timer2 + DT
        timer_global = timer_global + DT
        
        # choice
        c = choice(alpha_t, beta_t, alpha)

        # append choices 
        if rft >= 100:
            if c == 0:
                resp1 = np.append(resp1, timer_global)
            else:
                resp2 = np.append(resp2, timer_global)
    
        
        # check if reinforcer is set
        if (c==0) and (timer1 >= RFT_TIME1[rft_current[c]]):
            r = 1
            rft_current[c] += 1
            rft += 1
        elif (c==1) and (timer2 >= RFT_TIME2[rft_current[c]]):
            r = 1
            rft_current[c] += 1
            rft += 1
        else:
            r = 0

        # reset timer after reward
        if r == 1:
            #print("Get reward at " + str(timer1) + "s from " + str(c+1) + "th option")
            if c == 0:
                timer1 =  0
            else:
                timer2 = 0
        
        # update choice models
        alpha_t[c] = alpha_t[c] + r * 0.1
        beta_t[c] = beta_t[c] + (1-r)  * 0.1

        # append G
        alpha = math.exp(2 * lam)
        if rft >= 100:
            G_seq = np.vstack([G_seq, G(alpha_t, beta_t, alpha)])

        # break if all reinforcer is obtained
        if rft ==  NUM_RFT:
            break

    return G_seq, resp1, resp2


#  plotting abline
def abline(slope, intercept):
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')



# simulate stationary matching
'''
VI1 = 30
VI2 = 5
np.random.seed(34567)
G_seq, resp1, resp2 = sim_matching(VI1, VI2)
len(resp1)/(len(resp1) + len(resp2))
1-VI1/(VI1+VI2)
plt.plot(G_seq[1:, 0])
plt.plot(G_seq[1:, 1])
'''


df = pd.DataFrame({'VI1': [],
              'VI2': [],
              'resp_rate' : [],
              'rft_rate' : []})


VI1 = 30
for VI2 in np.array([1, 5, 10, 15, 30, 60, 90, 120, 240, 480]):
    G_seq, resp1, resp2 = sim_matching(VI1, VI2)
    rates = len(resp1)/(len(resp1) + len(resp2))
    rft_rate = 1-VI1/(VI1+VI2)
    temp = pd.DataFrame({'VI1': [VI1],
            'VI2': [VI2],
            'resp_rate' : [rates],
            'rft_rate' : [rft_rate]})
    df = pd.concat([df,temp])
    print( "conc VI " + str(VI1) + " VI " + str(VI2) + " has been done")



plt.scatter(df["rft_rate"], df["resp_rate"], s = 30)
abline(1, 0)





#plt.plot(G_seq[5000:, 0])
#plt.plot(G_seq[5000:, 1])
#plt.plot(G_seq[:, 0])
#plt.plot(G_seq[:, 1])


#print("response ratio: " + str(len(resp1)/(len(resp1) + len(resp2))) )
#print("reinforcement rates: " +  str(1-VI1/(VI1+VI2)) )
#print("final choice prob: " + str(softmax(-G_seq[len(G_seq)-1,:])))

#plt.plot(G_seq[11000:15000, 0])
#plt.plot(G_seq[11000:15000, 1])
# matching simulation (ongoing)

import numpy as np
from scipy.special import digamma, betaln
from scipy.special import betaincinv
import matplotlib.pyplot as plt
import pandas as pd
import math
from matching_functions import Agent, Environment_VI

# set config
VI1 = 30
VI2 = 60 #np.array([1, 5, 10, 15, 30, 60, 90, 120, 240, 480])
NUM_RFT = 2000
MINIMUM = 0.5
DT = 1

# define agent and environment, and initiaze them
agent = Agent()
env = Environment_VI()
agent.initialize_parameters(alpha=np.array([1., 1.]), beta=np.array([1., 1.]), \
mu=np.array([.5, .5]), eta=np.array([.0, .0]), omega=.0, lam=.3, a=1., b=20.)
env.initialize_enviroment(VI1, VI2, NUM_RFT, MINIMUM)


# store variables
rft = 0 
resp = []
resp_rft = []
alpha1 = []
alpha2 = []
beta1 = []
beta2 = []
mu1 = []
mu2 = []
eta1 = []
eta2 = []
a = []
b = []
m = []
prob_seq = []

# trial loop 
while True:

    # expected free energy
    G = agent.G(agent.alpha, agent.beta, agent.lam)
    
    # choice probability
    prob = agent.calc_action_prob(G)
    prob_seq.append(prob)

    # perform choice
    c = agent.take_action(prob)
    
    # if o = 1 (rewarded), update parameter accordingly
    o = env.return_observation(c)
    if o == 1:
        rft += 1
        if c==0:
            resp_rft.append(0)
        else:
            resp_rft.append(1)
    
    if rft == NUM_RFT:
        break
    
    # update parameters
    agent.update_parameters(o, c)

    # update time
    env.update_time(DT)
    env.reward_setting()



    # append variables
    resp.append(c)
    alpha1.append(agent.alpha[0])
    alpha2.append(agent.alpha[1])
    beta1.append(agent.beta[0])
    beta2.append(agent.beta[1])
    mu1.append(agent.mu[0])
    mu2.append(agent.mu[1])
    eta1.append(agent.eta[0])
    eta2.append(agent.eta[1])
    a.append(agent.a)
    b.append(agent.b)
    m.append(agent.m)


plt.plot(prob_seq)


print(sum(np.array(resp)==0)/len(resp))
1-VI1/ (VI1 + VI2)
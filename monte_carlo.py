from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

"""
input:
    env_spec: environment spec
    trajs: N trajectories generated using
        list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
    bpi: behavior policy used to generate trajectories
    pi: evaluation target policy
    initQ: initial Q values; np array shape of [nS,nA]
ret:
    Q: $q_pi$ function; numpy array shape of [nS,nA]
"""

def off_policy_mc_prediction_ordinary_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    
    nA = env_spec.nA        # Number of actions
    nS = env_spec.nS        # Number of states
      
    gamma = env_spec.gamma  # Gamma
    
    Q = np.zeros(shape=(nS, nA))
    C = np.zeros(shape=(nS, nA))
        
    for episode in range(len(trajs)):
        G = 0.0
        W = 1.0 
        
        for t in range(len(trajs[episode]))[::-1]:
            state, action, reward, next_state = trajs[episode][t]
            G = gamma * G + reward
            C[state, action] += 1
            
            # Update Q-values
            Q[state, action] += (W / C[state, action]) * (G - initQ[state, action])
           
            # Update the weights with ordinary importance sampling
            W = W * (pi.action_prob(state, action) / bpi.action_prob(state, action))
            
            if W == 0:
                break
        initQ = Q.copy() 
    return Q


"""
input:
    env_spec: environment spec
    trajs: N trajectories generated using behavior policy bpi
        list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
    bpi: behavior policy used to generate trajectories
    pi: evaluation target policy
    initQ: initial Q values; np array shape of [nS,nA]
ret:
    Q: $q_pi$ function; numpy array shape of [nS,nA]
"""

def off_policy_mc_prediction_weighted_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    
    nA = env_spec.nA        # Number of actions
    nS = env_spec.nS        # Number of states
      
    gamma = env_spec.gamma  # Gamma
    
    Q = np.zeros(shape=(nS, nA))
    C = np.zeros(shape=(nS, nA))
        
    for episode in range(len(trajs)):
        G = 0.0
        W = 1.0 
        
        for t in range(len(trajs[episode]))[::-1]:
            
            state, action, reward, next_state = trajs[episode][t]
            G = gamma * G + reward
            C[state, action] += W
            Q[state, action] += (W / C[state, action]) * (G - initQ[state, action])
            
            # Update weights with weighted importance sampling ratio
            W = W * (pi.action_prob(state, action) / bpi.action_prob(state, action))
            
            if W == 0:
                break
        initQ = Q.copy() 
    
    return Q

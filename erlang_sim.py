import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import genjax
from genjax import GenerativeFunction, ChoiceMap, Selection, trace
from genjax.generative_functions.distributions import TFPDistribution
from genjax.typing import Callable 
from genjax.inference.smc import *
from dataclasses import dataclass
from tensorflow_probability.substrates import jax as tfp
import IPython.core.completer
from scipy.stats import erlang
%matplotlib qt

# Looking at the basal ganglia
# expected value of importance weight is P(data) – there is a huge range on how likely the data. imagine flipping 100 times.
# sometimes your p(y) is bigger than your p(y) by 100000. if you're sending in P/Q it could be that there are no spikes – you need to do the norm on the multiplier.

# 1/Q count and a P count and a fixed denom for each. Kq and Kp. Multiplier neuron accepts 1/Q count and the P count membrane potential is a log of (1/Q) / Kq  + log(P) / Kp. Output. Exp on the membrane potential is your output. 


λ1 = 5
λ2 = 10
k = 5
key = jax.random.PRNGKey(np.random.randint(10000))
# note scale (beta)  is 1/λ
# in a gamma, alpha (concentration) is the same as k (shape) in an erlang. for a gamma to be an erlang, you just need an integer valued k, so that the sample is the time it takes to get to k events with a rate param of lambda. 
num_trials = 20
winner = []
erlang1 = tfp.distributions.Gamma(concentration=k, rate=jnp.log(λ1))
erlang2 = tfp.distributions.Gamma(concentration=k, rate=jnp.log(λ2))

if __name__ == "__main__":

    for i in range(num_trials):
        key, subkey = jax.random.split(key, 2)
        e1 = erlang1.sample(seed=subkey)
        key, subkey = jax.random.split(key, 2)
        e2 = erlang2.sample(seed=subkey)
        if e1 < e2:
            winner.append(1)
        else:
            winner.append(2)

    sns.histplot(winner)
    plt.show()

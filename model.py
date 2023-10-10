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

sns.set_theme(style="white")
console = genjax.pretty(width=80)
#key = jax.random.PRNGKey(314159)

sceneshape = [(0.0,10.0), (0.0,10.0)]
position_std = 1.0
    
truncnorm = lambda mu, sig: tfp.distributions.TruncatedNormal(mu, sig, 
                                                              sceneshape[0][0], sceneshape[0][1])

discrete_truncnorm = lambda mu, sig: tfp.distributions.QuantizedDistribution(truncnorm(mu, sig))

discrete_norm = TFPDistribution(discrete_truncnorm)

key, sub_key = jax.random.split(key)
tr = genjax.simulate(discrete_norm)(sub_key, (5.0, 1.0))











#    print(tr.get_choices()["x"])



# @dataclass
# class DeferredTFPDistribution:
#     func: Callable
#     def __call__(self, *args):
#         return self.func(*args)

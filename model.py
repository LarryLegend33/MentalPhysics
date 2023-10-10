import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import genjax
from genjax import GenerativeFunction, ChoiceMap, Selection, traceq
from genjax.generative_functions.distributions import TFPDistribution
from genjax.inference.smc import *
import tensorflow_probability as tfp

sns.set_theme(style="white")

console = genjax.pretty(width=80)

sceneshape = [(0.0,10.0), (0.0,10.0)]
position_std = 1.0

truncnorm = lambda mu, sig: tfp.distributions.TruncatedNormal(mu, sig,
                                                              sceneshape[0][0], sceneshape[0][1])

discrete_truncnorm = lambda mu, sig: tfp.distributions.QuantizedDistribution(truncnorm(mu, sig))

print(discrete_truncnorm(5.0, position_std).sample([5]))

discrete_norm = TFPDistribution(discrete_truncnorm)

key = jax.random.PRNGKey(314159)
key, sub_key = jax.random.split(key)
tr = genjax.simulate(discrete_norm)(sub_key, (5.0, 1.0))    



@genjax.gen
def trunctest(x_obs):
    x = genjax.tfp_truncated_normal(x_obs, 1.0, 0.0, 10.0) @ "x"
#    x = genjax.normal(x_obs, 2.0) @ "x"
    return x


if __name__ == "__main__":


#    print(tr.get_choices()["x"])




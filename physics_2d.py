import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import genjax
from genjax import GenerativeFunction, ChoiceMap, ExactDensity, Selection, trace
from genjax.generative_functions.distributions import TFPDistribution
from genjax.typing import Callable 
from genjax.inference.smc import *
from dataclasses import dataclass
from tensorflow_probability.substrates import jax as tfp
import IPython.core.completer
import pygame as pg
tfd = tfp.distributions
sns.set_theme(style="white")
console = genjax.pretty(width=80)

key = jax.random.PRNGKey(314159)
scene_dim = 10
σ_pos = 1.0
σ_vel = 1.0
maxvel = 4.0
positions = jnp.arange(0, scene_dim).astype(jnp.float32)
velocities = jnp.arange(-maxvel, maxvel+1).astype(jnp.float32)
truncwin = 2

""" Define Distributions """
    
@dataclass
class LabeledCategorical(ExactDensity):
    def sample(self, key, probs, labels, **kwargs):
        cat = tfd.Categorical(probs=probs)
        cat_index = cat.sample(seed=key)
        return labels[cat_index]

    def logpdf(self, v, probs, labels, **kwargs):
        w = jnp.sum(jnp.log(probs) * (labels==v))
        return w

class UniformCategorical(ExactDensity):
    def sample(self, key, labels, **kwargs):
        cat = tfd.Categorical(probs=jnp.ones(len(labels)) / len(labels))
        cat_index = cat.sample(seed=key)
        return labels[cat_index]

    def logpdf(self, v, labels, **kwargs):
        probs = jnp.ones(len(labels)) / len(labels)
        logpdf = jnp.log(probs)
        w = logpdf[0]
        return w

cat = TFPDistribution(lambda p: tfd.Categorical(probs=p))
labcat = LabeledCategorical()
uniformcat = UniformCategorical()
normalize = lambda x: x / jnp.sum(x)
discrete_norm = lambda μ, σ, dom: normalize(jnp.array([tfd.Normal(loc=μ, scale=σ).cdf(i + .5) - tfd.Normal(loc=μ, scale=σ).cdf(i - .5) for i in dom]))
truncate = lambda μ, dom, discnorm_vals: jnp.array([jax.lax.cond(abs(d-μ) <= truncwin, lambda: normval, lambda: 0.) for d, normval in zip(dom, discnorm_vals)])
discrete_truncnorm = lambda μ, σ, dom: normalize(truncate(μ, dom, discrete_norm(μ, σ, dom)))

""" Generative Functions for Model """

def vel_change_probs(velₚ, posₚ):
    if (posₚ <= positions[0] and velₚ < 0) or (posₚ >= positions[-1] and velₚ > 0):
        return discrete_truncnorm(-velₚ, σ_vel, velocities)
    else:
        return discrete_truncnorm(velₚ, σ_vel, velocities)

@genjax.gen
def init_latent_model():
    occ = uniformcat(positions) @ "occ"
    x = uniformcat(positions) @ "x"
    y = uniformcat(positions) @ "y"
    vx = uniformcat(velocities) @ "vx"
    vy = uniformcat(velocities) @ "vy"
    return (occ, x, y, vx, vy)

@genjax.gen
def invert_velocity(vₚ):
    v = labcat(discrete_truncnorm(-vₚ, σ_vel, velocities), velocities) @ "value"
    return v
    
@genjax.gen
def samesign_velocity(vₚ):
    v = labcat(discrete_truncnorm(vₚ, σ_vel, velocities), velocities) @ "value"
    return v

velocity_switch = genjax.Switch(invert_velocity, samesign_velocity)

@genjax.gen
def step_latent_model(occₚ, xₚ, yₚ, vxₚ, vyₚ):
    x_collision = jnp.array(
        ((xₚ <= positions[0]) * (vxₚ < 0)) + ((xₚ >= positions[-1]) * (vxₚ > 0)),
        dtype=int)
    y_collision = jnp.array(
        ((yₚ <= positions[0]) * (vyₚ < 0)) + ((yₚ >= positions[-1]) * (vyₚ > 0)),
        dtype=int)
    vx = velocity_switch(x_collision, vxₚ) @ "vx"
    vy = velocity_switch(y_collision, vyₚ) @ "vy"
    occ = cat(discrete_norm(occₚ, σ_pos, positions)) @ "occ"
    x = cat(discrete_truncnorm(xₚ + vxₚ, σ_pos, positions)) @ "x" 
    y = cat(discrete_truncnorm(yₚ + vyₚ, σ_pos, positions)) @ "y"
    return (occ, x, y, vx, vy)

    
key, subkey = jax.random.split(key, 2)
tr =genjax.simulate(step_latent_model)(subkey, (2., 4., 5., 1., 1.))
# Do we need to implement the recurse operator for step latent model here?

""" Renderer """

# there are multiple levels to this. there is a groundtruth square and occluder trace and a rendering via the obs model. the pixels are under the noise model (they can switch to other colors). for pygame, you just want to render out the pixels 100 times (i.e. draw 100 rectangles per loop). 


def make_random_obs():
    obs = []
    obslength = 20
    key = jax.random.PRNGKey(100)
    for i in range(obslength):
        key, subkey = jax.random.split(key, 2)
        random_frame = jax.random.randint(subkey, 
                                          minval=0, maxval=3, shape=(scene_dim, scene_dim))
        obs.append(random_frame)
    return obs

def render_physics(obs):
    # traverse a permutation first.
    colors = ["white", "blue", "red"]
    screenscale = 100
    pixsize = 1
    win = pg.display.set_mode((scene_dim*screenscale, scene_dim*screenscale))
    render = pg.Surface((scene_dim, scene_dim))
    running = True
    while running:
        for frame in obs:
            pg.time.wait(1000)
            render.fill("white")
            for i in range(scene_dim):
                for j in range(scene_dim):
                    for event in pg.event.get():
                        if event.type == pg.QUIT:
                            running = False
            
                    pg.draw.rect(render, colors[frame[i,j]], (i, j, pixsize, pixsize))
            win.blit(pg.transform.scale(render, win.get_rect().size), (0, 0))
            pg.display.update()
    pg.quit()
    



# Write the renderer. Input a trace, get out an animation in pygame. 
    

# OLD BUT USEFUL

# truncnorm = lambda μ, σ, dim: tfp.distributions.TruncatedNormal(μ, σ, dim[0], dim[1])

# discrete_truncnorm_tfp = lambda μ, σ, dim: tfp.distributions.QuantizedDistribution(truncnorm(μ, σ, dim))
# discrete_truncnorm = TFPDistribution(discrete_truncnorm_tfp)

# unif = lambda dim: tfp.distributions.Uniform(dim[0], dim[1])
# discrete_uniform_tfp = lambda dim: tfp.distributions.QuantizedDistribution(unif(dim))
# discrete_uniform = TFPDistribution(discrete_uniform_tfp) 
#key, subkey = jax.random.split(key)
#tr = genjax.simulate(discrete_truncnorm)(subkey, (5.0, 1.0, sceneshape[0]))

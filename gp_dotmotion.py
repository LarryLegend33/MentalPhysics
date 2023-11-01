# Example file showing a circle moving on screen
import pygame
import sys
import numpy as np
from graphviz import Digraph
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
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

# pygame setup
pygame.init()
screen_dims = (500., 500.)
env_dims = (20., 20.)
steps = screen_dims[0] / env_dims[0]

screen = pygame.display.set_mode(screen_dims)
clock = pygame.time.Clock()
running = True
dt = 0

length_sim = 200
index_points_ = np.random.uniform(0, env_dims[0], (length_sim, 1)).astype(np.float32)
observation_noise_var = 0.1
key = jax.random.PRNGKey(314159)

amplitude = 10
length_scale = .3
kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)
gp = lambda oip, onoise: tfd.GaussianProcess(kernel=kernel,
                                             index_points=oip,
                                             observation_noise_variance=onoise)
gp_jax = TFPDistribution(gp)
key, sub_key = jax.random.split(key, 2)
tr = genjax.simulate(gp_jax)(sub_key, (index_points_, observation_noise_var))

# Get julia code for probabilistic scene graph construction
# Also want to draw a categorical for motion type (Sin, ExpQuad, Lin)
# fully replicate the GP model here. discuss using the graphviz as a data structure (might have to use something different for jax -- but its completely doable just use nishad's framework if you want). 

class MotionGraph(Digraph):
    def __init__(self, dotmotion):
        self.dotmotion = dotmotion
        Digraph.__init__(self)


class GP_Graph(Digraph):
    def __init__(self):
        self.dotmotion = self.draw_GP()
        Digraph.__init__(self)

    def draw_GP(self):
        return 0 
        
        
def launch_anim(motion_scenegraph):
    # player_pos will be extracted from dotmotion using the scene graph. dotmotion
    # will contain the motion vectors x, y for a single sample from the gaussian process. 
    player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        # fill the screen with a color to wipe away anything from last frame
        screen.fill("black")

        if player_pos.x < dims[0]:
            player_pos.x += 1
        else:
            player_pos.x = 0

        pygame.draw.circle(screen, "white", player_pos, 5)
        # flip() the display to put your work on screen
        pygame.display.flip()

        # limits FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-
        # independent physics.
        dt = clock.tick(60) / 1000


    pygame.quit()



    # keys = pygame.key.get_pressed()
    # if keys[pygame.K_w]:
    #     player_pos.y -= 300 * dt
    # if keys[pygame.K_s]:
    #     player_pos.y += 300 * dt
    # if keys[pygame.K_a]:
    #     player_pos.x -= 300 * dt
    # if keys[pygame.K_d]:
    #     player_pos.x += 300 * dt
    # if keys[pygame.K_q]:
    #     running = False



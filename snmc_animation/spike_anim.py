from spike import Spike, Particle, SampleScore_RealTime, Resampler
import pygame as pg
import numpy as np
from scipy.stats import bernoulli

# will have a frame count. if a spike with that init time is present in a particle, 
# add it to the spike_queue.

# add to this loop the resampling step. currently, the sample score circuits return when they hit Kp and Kq. 

def render_snmc():
     # traverse a permutation first.
    spike_queue = []
    num_particles = 2
    num_latents_per_particle = 2
    num_states_per_latent = 5
    num_snmc_steps = 10
    scene_dim = 800
    length_simulation = 2000
    pixsize = 2
    win = pg.display.set_mode((scene_dim, scene_dim))
    particles = [Particle([SampleScore_RealTime(num_states_per_latent, win)
                           for n in range(num_latents_per_particle)],
                          600 - 300*p, 300) for p in range(num_particles)]
    for step in range(num_snmc_steps):
        for p in particles:
            spike_queue = p.step_samplescores(spike_queue)
            resampler = Resampler(particles, 200, 300)
            resampler.multiply_and_norm_idealized()
            resampler.extract_scoring_times()
  #            resampler.switch_states()

    win = pg.display.set_mode((scene_dim, scene_dim))
    running = True
    frame = 0
    while running:
        win.fill("black")
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
        pg.time.wait(100)
        for sp in spike_queue:
             sp.move_in_time(frame)
             sp.display(frame)
             if sp.spike_finished(frame):
                 spike_queue.remove(sp)
        pg.display.update()
        frame += 1
    pg.quit()
    return particles


# get the actual generation of spikes out of the classes. each class will just have spikes in time.
# you want to translate the spikes of each component to a Spike, but control the x and y position
# of spikes in the outer loop. 







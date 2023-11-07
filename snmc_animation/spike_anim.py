from spike import Spike, Particle, SampleScore_RealTime
import pygame as pg
import numpy as np
from scipy.stats import bernoulli

# will have a frame count. if a spike with that init time is present in a particle, 
# add it to the spike_queue.

# add to this loop the resampling step. currently, the sample score circuits return when they hit Kp and Kq. 

def render_snmc():
     # traverse a permutation first.
    spike_queue = []
    num_particles = 1
    scene_dim = 800
    length_simulation = 2000
    pixsize = 2
    win = pg.display.set_mode((scene_dim, scene_dim))
    running = True
    samplescores = [SampleScore_RealTime(5, win) for i in range(num_particles)]
    win = pg.display.set_mode((scene_dim, scene_dim))
    running = True
    frame = 0
    while running:
        win.fill("black")
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
        pg.time.wait(50)
        for sp in spike_queue:
             sp.move_in_time(frame)
             sp.display(frame)
             if sp.spike_finished(frame):
                 spike_queue.remove(sp)
        pg.display.update()
        frame += 1
    pg.quit()

# write a stripped down multiply and normalize here
# log(P / Kp) + log( 1/Q   / Kq)

def step_samplescores(particle, spike_queue):
    for s in particle.samplescores:
#        s.populate_assemblies()
        s.run_snmc()
        s.collect_spikes()
        spike_queue.extend(s.all_spikes)
    return spike_queue

def joint_score(particles):
# multiplication and norm should happen at the particle level.    

# based on particle scores, you'll switch states. state is represented in the particle.
# Resampler is a wrapper around particles; it takes scores and switches the state. 

# also code in the mental physics rendering 

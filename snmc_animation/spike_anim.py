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
    return particles, win


# get the actual generation of spikes out of the classes. each class will just have spikes in time.
# you want to translate the spikes of each component to a Spike, but control the x and y position
# of spikes in the outer loop. 


# this is actually very simple. scroll through the component list and 


def collect_spikes(obj, loc_x, loc_y, surface):

    spike_height = 5
    spacing = 2

    if type(obj) == SampleScore_RealTime:
        ss = obj
        spikes_in_timewin = lambda spikes: spikes[ss.t_lastscore:ss.t]
        # elements have to be arranged in y-order   
        wtas = [ss.wta[i] for i in range(ss.num_states)]
        assemblies = [ss.assemblies[pq + str(pq_ind)][n_id]
                      for pq in ["p", "q"] for n_id in range(
                       ss.neurons_per_assembly) for pq_ind in range(ss.num_states)]
        scoring = [ss.mux["p"]
                   ss.mux["q"],
                   ss.tik["p"],
                   ss.tik["q"],
                   ss.accum["q"]]
        elements = list(map(spikes_in_timewin, wtas + assemblies + scoring))
        sp_train_ylocs = range(loc_y, loc_y + (
            (spike_height + spacing) * len(elements)), spike_height+spacing)
        timewin = [ss.t_lastscore, ss.t]

    elif type(obj) == Resampler:
        resampler = obj
        timewin = resampler.time: resampler.time + len(resampler.spikes)
        resampler_spikes = []
        
    spikes_to_draw = []
    for i, spikelist in enumerate(elements):
        for ind, t in enumerate(timewin):
            if spikelist[ind] != 0:
                spikes_to_draw.append(Spike(loc_x, sp_train_ylocs[i], t, spike_height, surface))
    return spikes_to_draw

# need the right timer here




            

from spike import Spike, Particle, SampleScore, electrode
import pygame as pg
import numpy as np
from scipy.stats import bernoulli

# will have a frame count. if a spike with that init time is present in a particle, 
# add it to the spike_queue. 

def render_snmc():
     # traverse a permutation first.
    spike_queue = []
    num_particles = 1
    scene_dim = 800
    pixsize = 2
    v1_electrode = (490, 300, 100, 100)
    cp_electrode = (300, 200, 100, 100)
    gpi_electrode = (200, 200, 100, 100)
    lp_electrode = (200, 200, 100, 100)
    brain_location = (200, 300, 400, 400)
    pixsize = 2
    win = pg.display.set_mode((scene_dim, scene_dim))
    running = True
    brain = pg.image.load("MouseBrain.png")
#    electrode_r = loadImage("Electrode.png")
#    electrode_l = loadImage("Electrode_Reflected.png")

# quick hack here is having x and y coord of activity be member variables of the SampleScore class. 
    samplescores = [SampleScore(5, win) for i in range(num_particles)]
    for s in samplescores:
        s.populate_assemblies()
        s.run_snmc()
        s.populate_wta_and_scoring()
        spike_queue.extend(s.all_spikes)
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

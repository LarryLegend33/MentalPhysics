import math
import numpy as np
import pygame as pg


def electrode(x, y):
    return 1 
    
class Spike():
    
    def __init__(self, start_x, start_y, spike_init_time, spike_height, surface):
        self.start_x = start_x
        self.x = start_x
        self.y = start_y
        self.speed = 1
        self.init_time = spike_init_time
        self.spike_height = spike_height
        self.spike_stroke = "white"
        self.spike_strokeweight = 1
        self.end_time = 100
        self.surface = surface
        
    def move_in_time(self, fc):
        if self.init_time <= fc <= self.init_time + self.end_time: 
            self.x = self.x + self.speed
        
    def spike_finished(self, fc):
        return fc > self.init_time + self.end_time 
        
    def display(self, fc):
        if self.init_time <= fc <= self.init_time + self.end_time: 
            pg.draw.line(self.surface,
                         color=self.spike_stroke,
                         start_pos=(self.x, self.y),
                         end_pos=(self.x, self.y + self.spike_height),
                         width=self.spike_strokeweight)

# in Particle, you make a spike object for each element. 
# In the ElectrodeLocation class, parse according to P and Q? 

# try to make this flexible so that you could eventually have the 
# assemblies be generated on the fly. start with something very simple like a bernoulli draw.  

class Particle:
    def __init__(self, id, surface):
        self.length_of_simulation = 2000
        self.neurons_per_assembly = 2
        self.num_states = 2 
        self.kq = 5
        self.kp = 10
        # can make this amenable to larger assembly sizes and more latents. 
        # hand code for now. 
        self.assembly_indices = ["q0", "q1", "p0", "p1"]   
        self.assemblies = {i : [] for i in self.assembly_indices}
        self.mux = {"q": [], 
                    "p": []}
        self.tik = {"q": [], 
                    "p": []}
        self.accum = {"q": []}
        self.wta = {0: [],
                    1: []}      
        self.normalizer = []
        self.resampler = []
        self.all_spikes = []
        self.surface = surface
    
    def populate_assemblies(self):        
        sim_lambdas = [np.random.uniform(0, .2) for i in self.assembly_indices]
       # assembly_params = [s for s in sim_params for _ in (0, self.neurons_per_assembly)]
        for ai, λ  in zip(self.assembly_indices, sim_lambdas):
            self.assemblies[ai] = [poisson_process(λ, self.length_of_simulation)
                                for n in range(self.neurons_per_assembly)] 
                    
    def run_snmc(self):
        # have to implement counters too. 
        state = float("NaN")
        p_tik = 0
        q_tik = 0
        for t in range(self.length_of_simulation):
            if math.isnan(state):
                state = self.detect_winner(t)            
            if not math.isnan(state):
                qmux_spikes = self.find_spikes_in_assemblies(t, "q" + str(state))
                pmux_spikes = self.find_spikes_in_assemblies(t, "p" + str(state))
                self.wta[state].append(1)
                # when more than one state, just add a list here for all !state
                self.wta[not state].append(0)
                total_p_assembly_spikes = pmux_spikes + self.find_spikes_in_assemblies(t, "p" + str(int(not state)))
                total_q_assembly_spikes = qmux_spikes + self.find_spikes_in_assemblies(t, "q" + str(int(not state)))
                p_tik += total_p_assembly_spikes
                if p_tik <= self.kp:
                    self.mux["p"].append(pmux_spikes)
                    if p_tik == self.kp:
                        self.tik["p"].append(1)
                    else:
                        self.tik["p"].append(0)
                else: 
                    self.mux["p"].append(0)  
                    self.tik["p"].append(0)                     
                q_tik += qmux_spikes
                if q_tik <= self.kq:
                    self.mux["q"].append(qmux_spikes)
                    self.accum["q"].append(total_q_assembly_spikes)
                    if q_tik == self.kq:
                        self.tik["q"].append(1)
                    else:
                        self.tik["q"].append(0)                    
                else: 
                    self.mux["q"].append(0)
                    self.accum["q"].append(0)
                    self.tik["q"].append(0)
                if p_tik > self.kp and q_tik > self.kq:
                    p_tik = 0
                    q_tik = 0
                    state = float("NaN")
                                    
            else:
                self.mux["q"].append(0)
                self.mux["p"].append(0)
                self.wta[0].append(0)
                self.wta[1].append(0)
                self.tik["p"].append(0)
                self.tik["q"].append(0)
                self.accum["q"].append(0)
     
          
    def detect_winner(self, time):
        spike_in_0 = self.find_spikes_in_assemblies(time, "q0")
        spike_in_1 = self.find_spikes_in_assemblies(time, "q1")
        if spike_in_0 and not spike_in_1:
            return 0 
        elif spike_in_1 and not spike_in_0:
            return 1
        elif spike_in_1 and spike_in_0:
            return bernoulli(.5) 
        else:
            return float("NaN")
        
    def find_spikes_in_assemblies(self, time, assembly):
        spikes = sum([self.assemblies[assembly][i][time] for i in range(self.neurons_per_assembly)])
        return spikes
        
    def populate_wta_and_scoring(self):
    
        spike_height = 5
        spacing = 2
        cortex_x = 600
        cortex_y = 300
        # elements have to be arranged in y-order   
        wtas = [self.wta[0], self.wta[1]]
        assemblies = [self.assemblies[pq + str(pq_ind)][n_id] for pq in ["p", "q"] for n_id in range(
                       self.neurons_per_assembly) for pq_ind in range(self.num_states)]
        scoring = [self.mux["p"], self.mux["q"], self.tik["p"], self.tik["q"], self.accum["q"]]
        elements = wtas + assemblies + scoring 
        sp_train_ylocs = range(cortex_y, cortex_y + ((spike_height + spacing) * len(elements)), spike_height+spacing)
        for i, elem in enumerate(elements):
            self.assign_spikes(elem, cortex_x, sp_train_ylocs[i], spike_height)        
                    
    def assign_spikes(self, spikelist, x, y, spike_height):
        for t in range(self.length_of_simulation):
            if spikelist[t] != 0:
                self.all_spikes.append(Spike(x, y, t, spike_height, self.surface))            
        
bernoulli = lambda p: np.random.binomial(1, p)

def poisson_process(λ, num_timepoints):
    rate = λ * num_timepoints
    spikenum = np.random.poisson(rate)
    spiketimes = np.sort(num_timepoints * np.random.uniform(0, 1, spikenum)).astype(int)
    spiketrain = [1 if i in spiketimes else 0 for i in range(num_timepoints)] 
    return spiketrain
    





    
    
    

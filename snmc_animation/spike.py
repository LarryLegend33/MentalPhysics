import math
import numpy as np
import pygame as pg
import copy

normalize = lambda x: np.array(x) / sum(x)
    
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

#make sure to put 0 spike pads between t and tprev in the multiplier and resampler.

class Resampler:
    def __init__(self, particles, x, y):
        self.particles = particles
        self.particle_neuron_spikes = { p_id : [] for p_id in range(len(self.particles)) }
        self.x = x
        self.y = y
        self.t = float('nan')
        self.spikes = []
        self.time = self.extract_scoring_times()
        
    def multiply_and_norm_idealized(self):
        scores = []
        for p in self.particles:
            pscore = 0
            one_over_qscore = 0
            for s in p.samplescores:
                print("times")
                print(s.t_lastscore)
                print(s.t)
                pscore += np.log(sum(s.return_last_step("mux", "p")) / s.kp)
                one_over_qscore += np.log(sum(s.return_last_step("accum", "q")) / s.kq)
            scores.append(pscore + one_over_qscore)

        print("Scores")
        print(scores)
        normalizer_spikes = np.exp(scores)
        if sum(normalizer_spikes) == 0:
            # creates a uniform draw, but by now particle filter has collapsed. turn this into neural floating point as best as possible.
            normalizer_spikes += -1.0

        resampler_spikes = [np.random.multinomial(1, normalize(normalizer_spikes))
                            for p in self.particles]
        self.spikes = [normalizer_spikes, resampler_spikes]
        

    def switch_states(self):
        # eventually this will switch the wta states to using a different
        # particle as a parent, but haven't added the CPTs yet. CPTs will be a
        # metagraph structure 
        return 0

    def extract_scoring_times(self):
        sc_times = []
        for p in self.particles:
            for s in p.samplescores:
                sc_times.append(s.t)
        return np.max(sc_times)
                            
                            
        

class Particle:
    def __init__(self, samplescores, x, y):
        self.samplescores = samplescores
        self.x = x
        self.y = y
        
    def step_samplescores(self, spike_queue):
    # bump x or y here per samplescore to render in a different spot. 
        for s in self.samplescores:
           s.run_snmc()
           s.collect_spikes(self.x, self.y)
           spike_queue.extend(s.all_spikes)
        return spike_queue
        
# particle nests multiple SampleScores. Resampler takes particles.     

class SampleScore:
    def __init__(self, num_states, surface):

        self.neurons_per_assembly = 2
        self.num_states = num_states
        self.kq = 5
        self.kp = 10
        # can make this amenable to larger assembly sizes and more latents. 
        # hand code for now. 
        self.assembly_indices = ["q" + str(i) for i in range(self.num_states)] + [
            "p" + str(i) for i in range(self.num_states)]
        self.assemblies = {i : [[] for i in range(self.neurons_per_assembly)] for i in self.assembly_indices}
        self.sim_lambdas = [np.random.uniform(0, 1) for i in range(num_states)]
        self.mux = {"q": [], 
                    "p": []}
        self.tik = {"q": [], 
                    "p": []}
        self.accum = {"q": []}
        self.wta = { i : [] for i in range(self.num_states) } 
        self.all_spikes = []
        self.current_score = 0
        self.surface = surface
        self.component_dict = { "mux" : self.mux,
                                "tik" : self.tik,
                                "accum" : self.accum,
                                "wta" : self.wta,
                                "assemblies" : self.assemblies }
        
    def detect_winner(self, time):
        # turn this into a map over all q 
        spike_at_t = list(map(lambda x: self.find_spikes_in_assemblies(time, "q" + str(x)),
                              range(self.num_states)))
        if sum(spike_at_t) > 0:
            spiking_assemblies = np.nonzero(spike_at_t)[0]
            if len(spiking_assemblies) == 1:
                return spiking_assemblies[0]
            else:
                return np.random.randint(spiking_assemblies[0], spiking_assemblies[-1])
        else:
            return float("NaN")
        
    def find_spikes_in_assemblies(self, time, assembly):
        spikes = sum([self.assemblies[assembly][i][time] for i in range(self.neurons_per_assembly)])
        return spikes
        
  
# maybe a good idea here to just return the elements, and have assign spikes be outside the class. 
            
               
        

class SampleScore_RealTime(SampleScore):

    def __init__(self, num_states):
        self.state = float("NaN")
        self.t = 0
        self.t_lastscore = 0
        SampleScore.__init__(self, num_states)
        
    def poisson_spike(self):
        for assembly_neuron in self.assemblies.keys():
            # this indexes the assemblies using the labels qNUM or pNUM
            rate = self.sim_lambdas[int(assembly_neuron[1:])]
            for i in range(self.neurons_per_assembly):
                self.assemblies[assembly_neuron][i].append(np.random.poisson(rate, 1)[0])

    def return_last_step(self, component, p_or_q, *assembly_neuron_id):
        spiketrain = self.component_dict[component][p_or_q]
        if component == "assemblies" and assembly_neuron_id != ():
            return spiketrain[assembly_neuron_id[0]][self.t_lastscore:self.t]
        else:
            return spiketrain[self.t_lastscore:self.t]

    def run_snmc(self):
        # have to implement counters too.
        self.t_lastscore = copy.deepcopy(self.t)
        p_tik = 0
        q_tik = 0
        state = float("NaN")
        while True:
            self.poisson_spike()
            if math.isnan(state):
                state = self.detect_winner(self.t)            
            if not math.isnan(state):
                qmux_spikes = self.find_spikes_in_assemblies(self.t,
                                                             "q" + str(state))
                pmux_spikes = self.find_spikes_in_assemblies(self.t,
                                                             "p" + str(state))
                # this is just to space the wta spikes a bit              
                if np.random.random() > .6:
                    self.wta[state].append(1)
                else:
                    self.wta[state].append(0)
                # when more than one state, just add a list here for all !state
                for i in range(self.num_states):
                    if i != state:
                        self.wta[i].append(0)
                total_p_assembly_spikes = sum(map(
                    lambda x: self.find_spikes_in_assemblies(self.t, "p" + str(x)), range(self.num_states)))
                total_q_assembly_spikes = sum(map(
                    lambda x: self.find_spikes_in_assemblies(self.t, "q" + str(x)), range(self.num_states)))
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
                    self.state = state
                    return True
                    # END SIMULATION HERE, store the state in self.state
            else:
                self.mux["q"].append(0)
                self.mux["p"].append(0)
                for i in range(self.num_states):
                    self.wta[i].append(0)
                self.tik["p"].append(0)
                self.tik["q"].append(0)
                self.accum["q"].append(0) 
            self.t += 1

   
                
class SampleScore_Static(SampleScore):
    
    def __init__(self, num_states, surface):
        self.length_of_simulation = 2000
        SampleScore.__init__(self, num_states, surface)

    def populate_assemblies(self):        
        for ai in self.assembly_indices:
            self.assemblies[ai] = [poisson_process(self.sim_lambdas[int(ai[1:])],
                                                   self.length_of_simulation)
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
# this is just to space the wta spikes a bit – it looks like one big bar.                 
                if np.random.random() > .6:
                    self.wta[state].append(1)
                else:
                    self.wta[state].append(0)
                # when more than one state, just add a list here for all !state
                for i in range(self.num_states):
                    if i != state:
                        self.wta[i].append(0)
                total_p_assembly_spikes = sum(map(
                    lambda x: self.find_spikes_in_assemblies(t, "p" + str(x)), range(self.num_states)))
                total_q_assembly_spikes = sum(map(
                    lambda x: self.find_spikes_in_assemblies(t, "q" + str(x)), range(self.num_states)))
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
                for i in range(self.num_states):
                    self.wta[i].append(0)
                self.tik["p"].append(0)
                self.tik["q"].append(0)
                self.accum["q"].append(0)
     
    def start_snmc(self):
        self.populate_assemblies()
        self.run_snmc()
        self.populate_wta_and_scoring()
   
            
                
bernoulli = lambda p: np.random.binomial(1, p)

def poisson_process(λ, num_timepoints):
    rate = λ * num_timepoints
    spikenum = np.random.poisson(rate)
    spiketimes = np.sort(num_timepoints * np.random.uniform(0, 1, spikenum)).astype(int)
    spiketrain = [1 if i in spiketimes else 0 for i in range(num_timepoints)] 
    return spiketrain
    





    
    
    


import argparse
import numpy as np
import quantities
from brian2 import *
from neo.core import SpikeTrain
from elephant import conversion

#defaultclock.dt = 1*ms
directions = ['EE', 'EI', 'IE', 'II']

class mySNN():
    def __init__(self,
# network params
                 N=100,
                 inhibitory_ratio=0.2,
# neuron params
                 tau=30,
                 ref_period_exp=3, ref_period_inh=2, # in ms
                 rate=1.0, # in Hz
                 v_th=-54, v_rest=-74, v_reset=-74, # in mV
# synapse params
                 E_exc=0, E_inh=-100, # in mV
                 tau_ampa=2, tau_gaba=4, # in ms
                 delay_EE=1.5, delay_other=0.8, # in ms
                 w0=0.0,
                 gmax_exc=1.0, gmax_inh=1.0,
                 p_connection=1.0,
# short-term depression
                 U=0.4, delta_U=0.4,
                 tau_rec=150, # in ms
# excitatory STDP
                 beta=1.0,
                 tau_estdp=20, # in ms
                 delta_estdp=1.0e-3,
# inhibitory STDP
                 alpha=1.0,
                 tau_1=10, tau_2=20, # in ms
                 delta_istdp=1.0e-3,
# input layer configuration
                 input_strength=100.,
# record configuration
                 record=True,
                 record_state=False, record_synapse=False,
# initialize weights
                 init_weights=True):

# network params
        self.N = N
        self.N_exp = int(self.N*(1-inhibitory_ratio))
        self.N_inh = self.N-self.N_exp
        self.inhibitory_ratio = inhibitory_ratio

        delta_1 = delta_istdp/(1.-(alpha*tau_1/tau_2))
        delta_2 = delta_1*alpha*tau_1/tau_2
        self.params = {
            # neuron params
            'tau': tau*ms,
            'v_th': v_th*mV, 'v_rest': v_rest*mV, 'v_reset': v_reset*mV,
            # synapse params
            'E_exc': E_exc*mV, 'E_inh': E_inh*mV,
            'tau_ampa': tau_ampa*ms, 'tau_gaba': tau_gaba*ms,
            'w0': w0,
            'gmax_exc': gmax_exc, 'gmax_inh': gmax_inh,
            # short-term depression
            'U': U, 'delta_U': delta_U, 'tau_rec': tau_rec*ms,
            # excitatory STDP
            'tau_estdp': tau_estdp*ms,
            'delta_pre': delta_estdp, 'delta_post': beta*delta_estdp,
            # inhibitory STDP
            'tau_1': tau_1*ms, 'tau_2': tau_2*ms,
            'delta_1': delta_1, 'delta_2': delta_2,
        }
        self.freq_0 = rate*Hz*defaultclock.dt/np.exp((v_rest-v_th)/(4))
        self.en_b = 4*mV
        #self.freq_0 = 1.0
        #self.en_b = (v_rest-v_th)/np.log(rate*Hz*defaultclock.dt)*mV
        self.gmax_exc, self.gmax_inh = gmax_exc, gmax_inh
        self.p_connection = p_connection

        # LSM equations
        eqs_neuron = '''
        dv/dt = ((v_rest-v)+(E_exc-v)*gtot_exc+(E_inh-v)*gtot_inh)/tau : volt (unless refractory)
        dx/dt = (1-x)/tau_rec : 1
        rho = freq_0*exp((v-v_th)/en_b) : 1
        dgtot_exc/dt = -gtot_exc/tau_ampa : 1
        dgtot_inh/dt = -gtot_inh/tau_gaba : 1
        ref_period : second (constant)
        freq_0 : 1 (constant)
        en_b : volt (constant)
        '''
        #rho = freq_0*exp((v-v_th)/(4*mV)) : 1
        #rho = exp((v-v_th)/en_b) : 1
        #en_b : volt (constant)

        threshold = '(rand()<rho) and not_refractory'
        reset = '''
        v = v_reset
        x -= delta_U*x
        '''

        eqs_synapse_E = '''
        dapre/dt = -apre/tau_estdp : 1 (event-driven)
        dapost/dt = -apost/tau_estdp : 1 (event-driven)
        w : 1
        '''

        eqs_synapse_I = '''
        da1pre/dt = -a1pre/tau_1 : 1 (event-driven)
        da2pre/dt = -a2pre/tau_2 : 1 (event-driven)
        da1post/dt = -a1post/tau_1 : 1 (event-driven)
        da2post/dt = -a2post/tau_2 : 1 (event-driven)
        w : 1
        '''

        on_pre_E = '''
        gtot_exc_post += U*x_pre*w*gmax_exc
        apre += delta_pre
        w = clip(w+apost, 0, 1)
        '''

        on_post_E = '''
        apost -= delta_post
        w = clip(w+apre, 0, 1)
        '''

        on_pre_I = '''
        gtot_inh_post += U*x_pre*w*gmax_inh
        a1pre += delta_1
        a2pre += delta_2
        w = clip(w+a1post-a2post, 0, 1)
        '''

        on_post_I = '''
        a1post += delta_1
        a2post += delta_2
        w = clip(w+a1pre-a2pre, 0, 1)
        '''

        # Create LSM
        self.G = NeuronGroup(self.N, eqs_neuron, threshold=threshold, reset=reset, refractory='ref_period', method='euler', name='neurons')
        #self.G.v = '(v_th-v_rest)*rand()+v_rest'
        self.G.v = 'v_rest*mV'
        self.G.x = 1
        self.G.gtot_exc = 0
        self.G.gtot_inh = 0
        self.G.freq_0 = self.freq_0
        self.G.en_b = self.en_b
        self.G_exp = self.G[:self.N_exp]
        self.G_inh = self.G[self.N_exp:]
        self.G_exp.ref_period = ref_period_exp*ms
        self.G_inh.ref_period = ref_period_inh*ms

        self.S_EE = Synapses(self.G_exp, self.G_exp, eqs_synapse_E,
            on_pre=on_pre_E, on_post=on_post_E,
            delay=delay_EE*ms, method='euler',
            name='synapses_EE')
        self.S_EI = Synapses(self.G_exp, self.G_inh, eqs_synapse_E,
            on_pre=on_pre_E, on_post=on_post_E,
            delay=delay_other*ms, method='euler',
            name='synapses_EI')
        self.S_IE = Synapses(self.G_inh, self.G_exp, eqs_synapse_I,
            on_pre=on_pre_I, on_post=on_post_I,
            delay=delay_other*ms, method='euler',
            name='synapses_IE')
        self.S_II = Synapses(self.G_inh, self.G_inh, eqs_synapse_I,
            on_pre=on_pre_I, on_post=on_post_I,
            delay=delay_other*ms, method='euler',
            name='synapses_II')
        S_all = (self.S_EE, self.S_EI, self.S_IE, self.S_II)
        if init_weights:
            for direction, S in zip(directions, S_all):
                S.connect(condition='i!=j' if direction[0]==direction[1] else None, p=p_connection)
                S.w = w0

        self.input = SpikeGeneratorGroup(self.N, [], []*ms, name='input')
        self.S_input = Synapses(self.input, self.G,
            on_pre='v_post+=%f*mV'%input_strength, method='exact')
        self.S_input.connect(j='i')

        self.net = Network(
                    self.G,
                    self.S_EE, self.S_EI, self.S_IE, self.S_II,
                    self.input, self.S_input)

        self.spikemon_liquid = None
        self.statemon_liquid = None
        self.synapse_monitors = None
        self.record_synapse = record_synapse
        if record:
            self.spikemon_liquid = SpikeMonitor(self.G)
            self.net.add(self.spikemon_liquid)
        if record_state:
            self.statemon_liquid = StateMonitor(self.G, ['v', 'x', 'gtot_exc', 'gtot_inh'], record=True)
            self.net.add(self.statemon_liquid)
        if record_synapse & init_weights:
            self.synapse_monitors = {}
            for direction, S in zip(directions, S_all):
                self.synapse_monitors[direction] = StateMonitor(S, 'w', record=True)
                self.net.add(self.synapse_monitors[direction])

    def set_spikes(self, spikes_i, spikes_t):
        self.input.set_spikes(spikes_i, spikes_t)

    #def set_states(self, states):
    #    self.net.set_states(states)

    #def set_weights(self, params):
    #    for direction, S in zip(('EE', 'EI', 'IE', 'II'), (self.S_EE, self.S_EI, self.S_IE, self.S_II)):
    #        W = params['w_'+direction]
    #        sources, targets = np.where(np.isnan(W)==False)
    #        S.connect(i=sources, j=targets)
    #        S.w = W[sources, targets]
    #        S.x = 1
    #    var = ['x', 'w', 'g']
    #    if self.record_synapse & (self.synapse_monitors is None):
    #        self.synapse_monitors = {}
    #        for direction, S in zip(('EE', 'EI', 'IE', 'II'), (self.S_EE, self.S_EI, self.S_IE, self.S_II)):
    #            #var = ['x', 'w', 'g'] if direction[0]=='E' else ['x', 'w', 'g']
    #            self.synapse_monitors[direction] = StateMonitor(S, var, record=True)
    #            self.net.add(self.synapse_monitors[direction])

    #def initialize_with(self, states):
    def initialize_with(self, states, without_connect=False):
        self.G.v = states['v']*mV
        self.G.x = states['x']
        self.G.gtot_exc = states['gtot_exc']
        self.G.gtot_inh = states['gtot_exc']
        for d, S in zip(directions, (self.S_EE, self.S_EI, self.S_IE, self.S_II)):
            W = states['w_'+d]
            sources, targets = np.where(np.isnan(W)==False)
            if not without_connect: S.connect(i=sources, j=targets)
            #S.connect(i=sources, j=targets)
            S.w = W[sources, targets]
        self.S_EE.apre, self.S_EI.apre = states['apre_EE'], states['apre_EI']
        self.S_EE.apost, self.S_EI.apost = states['apost_EE'], states['apost_EI']
        self.S_IE.a1pre, self.S_II.a1pre = states['a1pre_IE'], states['a1pre_II']
        self.S_IE.a2pre, self.S_II.a2pre = states['a2pre_IE'], states['a2pre_II']
        self.S_IE.a1post, self.S_II.a1post = states['a1post_IE'], states['a1post_II']
        self.S_IE.a2post, self.S_II.a2post = states['a2post_IE'], states['a2post_II']

        if self.record_synapse & (self.synapse_monitors is None):
            self.synapse_monitors = {}
            for d, S in zip(directions, (self.S_EE, self.S_EI, self.S_IE, self.S_II)):
                self.synapse_monitors[d] = StateMonitor(S, 'w', record=True)
                self.net.add(self.synapse_monitors[d])

    def store(self, name='default', filename=None):
        self.net.store(name=name, filename=None)

    def restore(self, name='default', filename=None, restore_random_state=False):
        self.net.restore(name=name, filename=None, restore_random_state=restore_random_state)

    def run(self, runtime, name='default'):
        self.net.run(runtime, namespace=self.params)
        if self.spikemon_liquid is not None:
            return np.copy(self.spikemon_liquid.i), np.copy(self.spikemon_liquid.t/ms)*ms
        else: return None, None

    def run_without_output(self, runtime, name='default'):
        self.net.run(runtime, namespace=self.params)
        return None, None

    def get_spikes(self):
        return np.copy(self.spikemon_liquid.i), np.copy(self.spikemon_liquid.t/ms)*ms

    #def get_states(self, read_only_variables=True):
    #    return self.net.get_states()

    def get_binned_spike_trains(self, bin_size=20):
        t_stop = self.net.t/ms
        spike_trains = self.spikemon_liquid.spike_trains().values()

        binned_spike_trains = []
        for spike_train in spike_trains:
            st = SpikeTrain(spike_train/ms*quantities.ms, t_stop=t_stop*quantities.ms)
            binned_spike_trains.append(
                conversion.BinnedSpikeTrain(
                    st,
                    binsize=bin_size*quantities.ms,
                    t_start=0*quantities.ms).to_array()[0])
        bin_edges = np.arange(0, t_stop, bin_size)
        return bin_edges, np.array(binned_spike_trains)

    def get_liquid_voltage_trace(self):
        return self.statemon_liquid.v, self.statemon_liquid.t

    def get_state_monitor(self):
        return self.statemon_liquid

    def get_synapse_monitors(self):
        return self.synapse_monitors

    def get_weight_matrix(self, direction):
        if direction[0]=='E': n_sources=self.N_exp
        else: n_sources=self.N_inh
        if direction[1]=='E': n_targets=self.N_exp
        else: n_targets=self.N_inh

        W = np.full((n_sources, n_targets), np.nan)
        if direction=='EE':
            W[self.S_EE.i[:], self.S_EE.j[:]] = self.S_EE.w[:]
        elif direction=='EI':
            W[self.S_EI.i[:], self.S_EI.j[:]] = self.S_EI.w[:]
        elif direction=='IE':
            W[self.S_IE.i[:], self.S_IE.j[:]] = self.S_IE.w[:]
        else:
            W[self.S_II.i[:], self.S_II.j[:]] = self.S_II.w[:]
        return W

    #def get_params(self):
    #    param_dict = {
    #        "N": self.N, "N_exp": self.N_exp, "N_inh": self.N_inh,
    #        "inhibitory_ratio": self.inhibitory_ratio,
    #        "rate": self.rate,
    #        "gmax_exc": self.gmax_exc, "gmax_inh": self.gmax_inh,
    #        "p_connection": self.p_connection
    #        }
    #    for key, value in self.params.items():
    #        param_dict[key] = value
    #    return param_dict

    def save_weights(self, filename):
        Ws = {}
        for direction in ('EE', 'EI', 'IE', 'II'):
            Ws[direction] = self.get_weight_matrix(direction)
        np.savez(filename,
            w_EE=Ws['EE'], w_EI=Ws['EI'], w_IE=Ws['IE'], w_II=Ws['II'])

    def save_states(self, filename):
        Ws = {}
        for direction in ('EE', 'EI', 'IE', 'II'):
            Ws[direction] = self.get_weight_matrix(direction)
        np.savez(filename,
            v=self.G.v/mV,
            x=self.G.x,
            gtot_exc=self.G.gtot_exc,
            gtot_inh=self.G.gtot_inh,
            apre_EE=self.S_EE.apre, apre_EI=self.S_EI.apre,
            apost_EE=self.S_EE.apost, apost_EI=self.S_EI.apost,
            a1pre_IE=self.S_IE.a1pre, a1pre_II=self.S_II.a1pre,
            a2pre_IE=self.S_IE.a2pre, a2pre_II=self.S_II.a2pre,
            a1post_IE=self.S_IE.a1post, a1post_II=self.S_II.a1post,
            a2post_IE=self.S_IE.a2post, a2post_II=self.S_II.a2post,
            w_EE=Ws['EE'], w_EI=Ws['EI'], w_IE=Ws['IE'], w_II=Ws['II'])

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    #set_device('cpp_standalone')

    # Simulation setting
    runtime = 5*second
    np.random.seed(0)

    # Generate input spike trains
    #input_rate = 1*Hz
    #P = PoissonGroup(n_input, rates=input_rate)
    #MP = SpikeMonitor(P)
    #net = Network(P, MP)
    #net.run(runtime)
    #spikes_i = MP.i
    #spikes_t = MP.t

    snn = mySNN(N=100,
        inhibitory_ratio=0.2,
        record=True, record_state=True, record_synapse=True)
    #snn.set_spikes(spikes_i, spikes_t)

    liquid_spike_i, liquid_spike_t = snn.run(runtime)
    liquid_state_v, liquid_state_t = snn.get_liquid_voltage_trace()
    statemon_n = snn.get_state_monitor()
    statemon_s = snn.get_synapse_monitors()

    # Plot results
    neuron_id = 0

    fig, axs = plt.subplots(nrows=4, figsize=(12, 7), sharex=True)
    axs[0].plot(liquid_state_t/second, liquid_state_v[neuron_id]/mV)
    for t in liquid_spike_t[liquid_spike_i==neuron_id]/second:
        axs[0].axvline(t, ls='--', c='C1', lw=0.5)
    axs[0].set_ylabel('v [mV]')
    axs[1].plot(statemon_n.t/second, statemon_n[neuron_id].x)
    axs[1].set_ylabel('x []')
    axs[2].plot(statemon_n.t/second, statemon_n[neuron_id].gtot_exc)
    axs[2].set_ylabel('gtot_exc []')
    axs[3].plot(statemon_n.t/second, statemon_n[neuron_id].gtot_inh)
    axs[3].set_ylabel('gtot_inh []')
    axs[3].set_xlim(0, runtime/second)

    fig, axs = plt.subplots(nrows=2, figsize=(12, 7), sharex=True)
    bin_size = 20
    bin_edges, binned_spike_trains = snn.get_binned_spike_trains(bin_size=bin_size)
    axs[0].plot((bin_edges+bin_size/2)/1000., np.sum(binned_spike_trains, axis=0), c='k')
    axs[0].set_ylabel('Spike counts')
    axs[0].set_xlim(0, runtime/second)
    try:
        for t in spikes_t/second:
            axs[1].axvline(t, ls='-', c='r', alpha=0.2, lw=1)
    except: pass
    axs[1].plot(liquid_spike_t/second, liquid_spike_i, '.k', ms=1)
    axs[1].set_ylabel('Neuron index')
    axs[1].set_xlabel('Time, second')
    axs[1].set_xlim(0, runtime/second)

    for direction in ('EE', 'EI', 'IE', 'II'):
        print(direction, statemon_s[direction].w.shape)
        fig, axs = plt.subplots(nrows=2, figsize=(12, 3), sharex=True)
        trace = statemon_s[direction].w
        im = axs[0].imshow(trace, aspect='auto', origin='lower')
        axs[1].plot(np.mean(trace, axis=0), c='k', lw=1)
        axs[1].set_xlim(0, trace.shape[1])
        fig.colorbar(im, ax=axs)
        plt.suptitle(direction)


    #fig, axs = plt.subplots(nrows=2, figsize=(5, 5))
    #hist, bin_edges = np.histogram(np.sum(binned_spike_trains, axis=0), bins=100)
    ##print hist
    ##print bin_edges
    #hist = hist[1:]
    #bin_edges = bin_edges[1:-1]
    #axs[0].plot(bin_edges, hist, c='k')
    #axs[0].set_xscale('log')
    #axs[0].set_yscale('log')
    #axs[1].plot(bin_edges, hist, c='k')

    #fig, axs = plt.subplots(nrows=6, figsize=(12, 7), sharex=True)
    #axs[0].plot(liquid_state_t/ms, liquid_state_v[neuron_id]/mV)
    #for t in liquid_spike_t[liquid_spike_i==neuron_id]/ms:
    #    axs[0].axvline(t, ls='--', c='C1', lw=0.5)
    #axs[0].set_ylabel('v [mV]')
    #axs[0].set_xlim(0, runtime/ms)
    #for trace in s_monitor_EE[synapses].x:
    #    axs[1].plot(s_monitor_EE.t/ms, trace)
    #axs[1].set_ylabel('x []')
    #axs[1].set_xlim(0, runtime/ms)
    #for trace in s_monitor_EE[synapses].c:
    #    axs[2].plot(s_monitor_EE.t/ms, trace)
    #axs[2].set_ylabel('c []')
    #axs[2].set_xlim(0, runtime/ms)
    #for trace in s_monitor_EE[synapses].w:
    #    axs[3].plot(s_monitor_EE.t/ms, trace)
    #axs[3].set_ylabel('w []')
    #axs[3].set_xlim(0, runtime/ms)
    #bin_size = 20
    #bin_edges, binned_spike_trains = snn.get_binned_spike_trains(bin_size=bin_size)
    #axs[4].plot(bin_edges+bin_size/2, np.sum(binned_spike_trains, axis=0), c='k')
    #axs[4].set_ylabel('Neuron index')
    #axs[4].set_xlim(0, runtime/ms)
    #try:
    #    for t in spikes_t/ms:
    #        axs[5].axvline(t, ls='-', c='r', alpha=0.2, lw=1)
    #except: pass
    #axs[5].plot(liquid_spike_t/ms, liquid_spike_i, '.k', ms=1)
    #axs[5].set_ylabel('Neuron index')
    #axs[5].set_xlabel('Time [ms]')
    #axs[5].set_xlim(0, runtime/ms)
    plt.show()


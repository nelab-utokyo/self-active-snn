
import numpy as np
import quantities
from brian2 import *
from neo.core import SpikeTrain
from elephant import conversion
from utils import truncnorm

class myLSM():
    def __init__(self,
                 nx = 15, ny = 3, nz = 3, inhibitory_ratio = 0.2,
                 tau = 30*ms, tau_syne = 3*ms, tau_syni = 6*ms,
                 i_offset = 13.5*nA, noise_strength = 0.,
                 R_mem = 1*Mohm,
                 v_th=15*mV, v_reset = 13.5*mV,
                 lmd = 2,
                 C_EE = 0.3, C_EI = 0.2, C_IE = 0.4, C_II = 0.1,
                 U_EE = 0.5, U_EI = 0.05, U_IE = 0.25, U_II = 0.32,
                 D_EE = 1.1, D_EI = 0.125, D_IE = 0.7, D_II = 0.144,
                 F_EE = 0.05, F_EI = 1.2, F_IE = 0.02, F_II = 0.06,
                 A_EE = 30, A_EI = 60, A_IE = -19, A_II = -19,
                 delay_EE = 1.5*ms, delay_other = 0.8*ms,
                 n_input=1, prob_input2middle=0.3,
                 statemon_liquid=False, synapse_monitors=False):

        # LSM parameters
        N = nx*ny*nz
        N_exp, N_inh = int(N*(1-inhibitory_ratio)), int(N*inhibitory_ratio)
        C_all = [C_EE, C_EI, C_IE, C_II]
        U_means = [U_EE, U_EI, U_IE, U_II]
        D_means = [D_EE, D_EI, D_IE, D_II]
        F_means = [F_EE, F_EI, F_IE, F_II]
        A_means = [A_EE, A_EI, A_IE, A_II]
        self.neuron_params = {
            'tau': tau,
            'tau_syne': tau_syne,
            'tau_syni': tau_syni,
            'i_offset': i_offset,
            'noise_strength' : noise_strength*nA,
            'R_mem': R_mem,
            'v_th': v_th,
            'v_reset': v_reset
        }

        # LSM equations
        eqs_neuron = '''
        noise : 1 (linked)
        dv/dt = (-v + R_mem*(i_e + i_i + i_offset + noise_strength*noise))/tau : volt (unless refractory)
        di_e/dt = -i_e/tau_syne : amp
        di_i/dt = -i_i/tau_syni : amp
        ref_period : second
        x : 1
        y : 1
        z : 1
        '''

        eqs_noise = 'noise : 1' if noise_strength == 0. else 'dnoise/dt = -noise/tau + tau**-0.5*xi : 1'

        eqs_synapse = '''
        ddelta/dt = 1 : second (event-driven)
        u : 1
        R : 1
        U : 1
        D : second
        F : second
        A : amp
        '''

        on_pre_E = '''
        i_e_post += A*u*R
        u = U+u*(1-U)*exp(-delta/F)
        R = 1+(R-u*R-1)*exp(-delta/D)
        delta = 0*second
        '''

        on_pre_I = '''
        i_i_post += A*u*R
        u = U+u*(1-U)*exp(-delta/F)
        R = 1+(R-u*R-1)*exp(-delta/D)
        delta = 0*second
        '''

        # Create LSM
        self.G = NeuronGroup(N, eqs_neuron, threshold='v>v_th', reset='v=v_reset', refractory='ref_period', method='euler', name='neurons')
        self.G.v = '(v_th-v_reset)*rand()+v_reset'
        self.G_exp = self.G[:N_exp]
        self.G_inh = self.G[N_exp:]
        self.G_exp.ref_period = 3*ms
        self.G_inh.ref_period = 2*ms
        pos = np.array([np.arange(N)%nx, np.arange(N)/nx%ny, np.arange(N)/(nx*ny)]).T
        np.random.shuffle(pos)
        self.G.x = pos[:, 0]
        self.G.y = pos[:, 1]
        self.G.z = pos[:, 2]

        self.noise = NeuronGroup(N, eqs_noise, method='euler', name='noise')
        self.G.noise = linked_var(self.noise, 'noise', index=range(N))

        self.S_EE = Synapses(self.G_exp, self.G_exp, eqs_synapse, on_pre=on_pre_E, method='exact')
        self.S_EI = Synapses(self.G_exp, self.G_inh, eqs_synapse, on_pre=on_pre_E, method='exact')
        self.S_IE = Synapses(self.G_inh, self.G_exp, eqs_synapse, on_pre=on_pre_I, method='exact')
        self.S_II = Synapses(self.G_inh, self.G_inh, eqs_synapse, on_pre=on_pre_I, method='exact')
        S_all = [self.S_EE, self.S_EI, self.S_IE, self.S_II]
        for S, C, U_mean, D_mean, F_mean, A_mean in zip(S_all, C_all, U_means, D_means, F_means, A_means):
            S.connect(condition='i!=j', p='%f*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/(%d**2))'%(C, lmd))
            S.R = 1
            S.U = truncnorm(U_mean, U_mean/2., len(S.U))
            S.u = S.U
            S.D = truncnorm(D_mean, D_mean/2., len(S.D))*second
            S.F = truncnorm(F_mean, F_mean/2., len(S.F))*second
            if A_mean>0: S.A = truncnorm(A_mean, abs(A_mean), len(S.A))*nA
            else: S.A = truncnorm(A_mean, abs(A_mean), len(S.A), vmin=-np.inf, vmax=0)*nA
            S.delta = 0.1*second
        self.S_EE.delay = delay_EE
        self.S_EI.delay = delay_other
        self.S_IE.delay = delay_other
        self.S_II.delay = delay_other

        self.spikemon_liquid = SpikeMonitor(self.G)
        if statemon_liquid: self.statemon_liquid = StateMonitor(self.G, 'v', record=True)
        if synapse_monitors:
            self.synapse_monitors = {}
            for direction, S in zip(['EE', 'EI', 'IE', 'II'], S_all):
                self.synapse_monitors[direction] = StateMonitor(S, ['u', 'R'], record=True)

        self.input = SpikeGeneratorGroup(n_input, [], []*ms, name='input')
        self.S_input_E = Synapses(self.input, self.G_exp, on_pre='i_e_post+=18*nA', method='exact')
        self.S_input_I = Synapses(self.input, self.G_inh, on_pre='i_e_post+=9*nA', method='exact')
        self.S_input_E.connect(p=prob_input2middle)
        self.S_input_I.connect(p=prob_input2middle)

        self.net = Network(
                    self.G,
                    self.noise,
                    self.S_EE, self.S_EI, self.S_IE, self.S_II,
                    self.input, self.S_input_E, self.S_input_I,
                    self.spikemon_liquid)
        if statemon_liquid: self.net.add(self.statemon_liquid)
        if synapse_monitors:
            for direction in ['EE', 'EI', 'IE', 'II']:
                self.net.add(self.synapse_monitors[direction])

    def set_spikes(self, spikes_i, spikes_t):
        self.input.set_spikes(spikes_i, spikes_t)

    def set_states(self, values):
        self.net.set_states(values)

    def store(self, name='default', filename=None):
        self.net.store(name=name, filename=None)

    def restore(self, name='default', filename=None, restore_random_state=False):
        self.net.restore(name=name, filename=None, restore_random_state=restore_random_state)

    def run(self, runtime, name='default'):
        self.net.run(runtime, namespace=self.neuron_params)
        return np.copy(self.spikemon_liquid.i), np.copy(self.spikemon_liquid.t/ms)*ms

    def get_states(self):
        return self.net.get_states()

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

    def get_synapse_monitors(self):
        return self.synapse_monitors

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Simulation setting
    runtime = 1000*ms
    np.random.seed(0)

    # Generate input spike trains
    n_input = 100
    input_rate = 1*Hz
    P = PoissonGroup(n_input, rates=input_rate)
    MP = SpikeMonitor(P)
    net = Network(P, MP)
    net.run(runtime)
    spikes_i = MP.i
    spikes_t = MP.t

    lsm = myLSM(n_input=n_input, statemon_liquid=True, synapse_monitors=True, noise_strength=0)
    lsm.set_spikes(spikes_i, spikes_t)

    liquid_spike_i, liquid_spike_t = lsm.run(runtime)
    liquid_state_v, liquid_state_t = lsm.get_liquid_voltage_trace()
    s_monitor_EE = lsm.get_synapse_monitors()['EE']

    # Plot results
    neuron_id = 0
    synapse_id = 0

    fig, axs = plt.subplots(nrows=6, figsize=(12, 7), sharex=True)
    axs[0].plot(liquid_state_t/ms, liquid_state_v[neuron_id]/mV)
    for t in liquid_spike_t[liquid_spike_i==neuron_id]/ms:
        axs[0].axvline(t, ls='--', c='C1', lw=1)
    axs[0].set_ylabel('v [mV]')
    axs[1].plot(s_monitor_EE.t/ms, s_monitor_EE.u[synapse_id])
    axs[1].set_ylabel('u []')
    axs[2].plot(s_monitor_EE.t/ms, s_monitor_EE.R[synapse_id])
    axs[2].set_ylabel('R []')
    axs[3].plot(s_monitor_EE.t/ms, s_monitor_EE.u[synapse_id]*s_monitor_EE.R[synapse_id])
    axs[3].set_ylabel('u*R []')
    bin_size = 20
    bin_edges, binned_spike_trains = lsm.get_binned_spike_trains(bin_size=bin_size)
    axs[4].plot(bin_edges+bin_size/2, np.sum(binned_spike_trains, axis=0), c='k')
    axs[4].set_ylabel('Neuron index')
    for t in spikes_t/ms:
        axs[5].axvline(t, ls='-', c='r', alpha=0.2, lw=1)
    axs[5].plot(liquid_spike_t/ms, liquid_spike_i, '.k', ms=1)
    axs[5].set_ylabel('Neuron index')
    axs[5].set_xlabel('Time [ms]')
    plt.show()


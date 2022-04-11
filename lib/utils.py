
import numpy as np
from brian2 import *
from sklearn import linear_model

def truncnorm(mean, SD, size, vmin=0, vmax=np.inf):
    vals = np.random.normal(loc=mean, scale=SD, size=size)
    while 1:
        ids = np.where(np.logical_or(vals<=vmin, vals>=vmax))[0]
        #print mean, SD, size, vals[ids]
        if len(ids)==0: break
        else: vals[ids] = np.random.normal(loc=mean, scale=SD, size=len(ids))
    return vals

def jitter(spikes_i, spikes_t, time_max, time_min=0*ms, scale=100):
    jittered_i = []
    jittered_t = []
    if scale == 0:
        jittered_i = spikes_i
        jittered_t = spikes_t
    else:
        noise = np.random.normal(loc=0, scale=scale, size=len(spikes_t))
        jittered_t = spikes_t/ms + noise
        indices = np.argsort(jittered_t)
        jittered_t = jittered_t[indices]
        jittered_i = spikes_i[indices]

        jittered_t = np.unique((jittered_t*10).astype(np.int32).tolist()).astype(np.float32)*0.1*ms
        flag = np.logical_and(jittered_t>=time_min, jittered_t<time_max)
        jittered_i = spikes_i[flag]
        jittered_t = jittered_t[flag]
    return jittered_i, jittered_t

def van_Rossum_distance(st0_i, st0_t, st1_i, st1_t, tau=5*ms):
    tau_samples = int(tau/defaultclock.dt)
    exp_filter = np.exp(-np.arange(10*tau_samples)/tau_samples)
    st0_s = (st0_t/defaultclock.dt).astype(np.int32)
    st1_s = (st1_t/defaultclock.dt).astype(np.int32)
    n_indices = max(max(st0_i), max(st1_i))+1
    n_samples = max(max(st0_s), max(st1_s))+1
    st0_mat = np.zeros((n_indices, n_samples))
    st1_mat = np.zeros((n_indices, n_samples))
    for i, s in zip(st0_i, st0_s): st0_mat[i, s] = 1
    for i, s in zip(st1_i, st1_s): st1_mat[i, s] = 1
    l2norms = []
    for i in range(n_indices):
        x0 = np.convolve(st0_mat[i], exp_filter)
        x1 = np.convolve(st1_mat[i], exp_filter)
        l2norms.append(np.linalg.norm(x0-x1))
    dist = np.mean(l2norms)/np.sqrt(tau_samples)
    return dist

def spike_distance(st0_i, st0_t, st1_i, st1_t, tau=5*ms):
    tau_samples = int(tau/defaultclock.dt)
    exp_filter = np.exp(-np.square((np.arange(6*tau_samples)-3*tau_samples)/tau_samples))
    st0_s = (st0_t/defaultclock.dt).astype(np.int32)
    st1_s = (st1_t/defaultclock.dt).astype(np.int32)
    n_indices = max(max(st0_i), max(st1_i))+1
    n_samples = max(max(st0_s), max(st1_s))+1
    st0_mat = np.zeros((n_indices, n_samples))
    st1_mat = np.zeros((n_indices, n_samples))
    for i, s in zip(st0_i, st0_s): st0_mat[i, s] = 1
    for i, s in zip(st1_i, st1_s): st1_mat[i, s] = 1
    l2norms = []
    for i in range(n_indices):
        x0 = np.convolve(st0_mat[i], exp_filter)
        x1 = np.convolve(st1_mat[i], exp_filter)
        l2norms.append(np.linalg.norm(x0-x1))
    dist = np.mean(l2norms)
    return dist

def liquid_distance(x_u, x_v):
    return np.linalg.norm(x_u-x_v, axis=0)

def calc_cv(spike_t, spikes_i, N):
    cvs = []
    for i in range(N):
        indices = np.where(spikes_i==i)[0]
        ISIs = np.diff(spike_t[indices]/ms)
        m = np.mean(ISIs)
        s = np.std(ISIs)
        cvs.append(s/m)
    return cvs

#class low_pass_filter():
#    def __init__(self, N, tau=30*ms, R_mem=1*Mohm):
#        self.params = {
#            'tau': tau,
#            'R_mem': R_mem
#        }
#        eqs = '''
#        dv/dt = -v/tau : volt
#        '''
#        on_pre = 'v_post += 1*mV'
#        self.input = SpikeGeneratorGroup(N, [], []*ms, name='input')
#        self.LPF = NeuronGroup(N, eqs, method='exact', name='LPF')
#        self.S = Synapses(self.input, self.LPF, on_pre=on_pre)
#        self.S.connect(j='i')
#        self.statemon = StateMonitor(self.LPF, 'v', record=True)
#        self.net = Network(self.input, self.LPF, self.S, self.statemon)
#
#    def set_spikes(self, spikes_i, spikes_t):
#        self.input.set_spikes(spikes_i, spikes_t)
#
#    def store(self, name='default', filename=None):
#        self.net.store(name=name, filename=None)
#
#    def restore(self, name='default', filename=None, restore_random_state=False):
#        self.net.restore(name=name, filename=None,
#            restore_random_state=restore_random_state)
#
#    def run(self, runtime, name='default'):
#        self.net.run(runtime, namespace=self.params)
#        return np.copy(self.statemon.v), np.copy(self.statemon.t/ms)*ms

DEFAULTCLOCK = defaultclock.dt/second
def low_pass_filter(spikes_i, spikes_t, N, runtime, tau=0.03):
    l_kernel = int(tau/DEFAULTCLOCK)*10
    l_runtime = int(runtime/DEFAULTCLOCK)
    kernel = np.exp(-np.arange(l_kernel)/float(l_kernel)*10)
    voltage = np.zeros((N, l_kernel+l_runtime), dtype=np.float64)
    times = (spikes_t/DEFAULTCLOCK).astype(np.int64)
    trace_t = np.arange(l_runtime, dtype=np.float64)*DEFAULTCLOCK
    trace_v = np.zeros((N, l_runtime), dtype=np.float64)

    for i, t in zip(spikes_i, times):
        voltage[i, t:t+l_kernel] += kernel

    trace_v = voltage[:, :l_runtime]
    return trace_v, trace_t

def linear_regression(x, y, x_lreg):
    y_emp = np.zeros((x_lreg.shape[0], 1))
    for i, v in enumerate(x_lreg[:, 0]):
        if v in x:
            y_emp[i, 0] = y[np.where(x==v)[0][0]]

    min_error = np.inf
    clf = linear_model.LinearRegression()
    y_best = None
    range_best = None
    for i in range(6, len(x)):
        clf.fit(x[1:i], y[1:i])
        #indices = np.where(x_lreg<=x[i-1])[0]
        indices = np.where((x_lreg>0)&(x_lreg<=x[i-1]))[0]
        y_lreg = clf.predict(x_lreg[indices])
        #error = np.mean(np.square(y_lreg-y_emp[indices]))
        #error = np.abs(np.sum(y_lreg-y_emp[indices]))
        error = np.sum(np.square(y_lreg-y_emp[indices]))/(len(indices)-1)/np.sqrt(len(indices))
        #print(y_lreg.reshape(-1))
        #print(y_emp[indices].reshape(-1))
        if error <= min_error:
            #print(i, error)
            y_best = clf.predict(x_lreg)
            range_best = [1, i]
            min_error = error
    return y_emp, y_best, range_best


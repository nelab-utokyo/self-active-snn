
import numpy as np
from brian2 import second, defaultclock
cimport cython
cimport numpy as np

cdef float DEFAULTCLOCK = defaultclock.dt/second

cdef void _convolve(
        np.ndarray[np.int32_t, ndim=1] spikes_i, np.ndarray[np.int64_t, ndim=1] times,
        np.ndarray[np.float64_t, ndim=2] voltage,
        int l_kernel,
        np.ndarray[np.float64_t, ndim=1] kernel):

    cdef int i, t
    cdef int i_max = spikes_i.shape[0]

    for i in range(i_max):
        t = times[i]
        voltage[spikes_i[i], t:t+l_kernel] += kernel

cdef class low_pass_filter:
    cdef public int N
    cdef public float tau
    cdef public int l_kernel
    cdef np.ndarray kernel

    def __init__(self, int N, float tau=0.03):
        self.N = N
        self.tau = tau
        self.l_kernel = int(tau/DEFAULTCLOCK)*10
        self.kernel = np.exp(-np.arange(self.l_kernel)/float(self.l_kernel)*10)

    def filt(
            self, 
            np.ndarray[np.int32_t, ndim=1] spikes_i, np.ndarray[np.float64_t, ndim=1] spikes_t,
            float runtime):
        
        cdef int l_runtime = int(runtime/DEFAULTCLOCK)
        cdef np.ndarray[np.float64_t, ndim=2] voltage = np.zeros((self.N, self.l_kernel+l_runtime), dtype=np.float64)
        cdef np.ndarray[np.int64_t, ndim=1] times = (spikes_t/DEFAULTCLOCK).astype(np.int64)
        cdef np.ndarray[np.float64_t, ndim=1] trace_t = np.arange(l_runtime, dtype=np.float64)*DEFAULTCLOCK
        cdef np.ndarray[np.float64_t, ndim=2] trace_v = np.zeros((self.N, l_runtime), dtype=np.float64)

        _convolve(spikes_i, times, voltage, self.l_kernel, self.kernel)
        trace_v = voltage[:, :l_runtime]
        return trace_v, trace_t

@cython.boundscheck(False)
@cython.wraparound(False)
def _pdelta(
        np.ndarray[np.float64_t, ndim=3] weights,
        np.ndarray[np.float64_t, ndim=2] trace_v,
        np.ndarray[np.int32_t, ndim=1] d,
        float eta, float mu, float epsilon, float gamma, float threshold):

    cdef int i, j, cls
    cdef int n_classes = d.shape[0]
    cdef int n_samples = trace_v.shape[0]
    cdef int n_perceptrons = weights.shape[2]
    cdef float error = 0
    cdef float w_ix, o, o_
    cdef np.ndarray[np.float64_t, ndim=2] v_perceptron
    cdef np.ndarray[np.float64_t, ndim=1] O_
    cdef np.ndarray[np.float64_t, ndim=1] delta, x

    #o = d[cls]
    #x = trace_v[j]
    #wx = v_perceptron[j]
    #o_ = O_[j]
    #w_ix = wx[i]
    for cls in range(n_classes):
        o = d[cls]
        v_perceptron = np.dot(trace_v, weights[cls])
        O_ = np.sum(v_perceptron>15, axis=1)/float(v_perceptron.shape[1])
        for j in range(n_samples):
            x = trace_v[j]
            o_ = O_[j]
            for i in range(n_perceptrons):
                w_ix = v_perceptron[j][i]
                if o_>o+epsilon and w_ix>=0:
                    delta = -eta*x
                elif o_<o-epsilon and w_ix<0:
                    delta = eta*x
                elif o_<=o+epsilon and 0<=w_ix and w_ix<gamma:
                    delta = eta*mu*x
                elif o_>=o-epsilon and -gamma<w_ix and w_ix<0:
                    delta = -eta*mu*x
                else:
                    continue
                weights[cls][:, i] += delta
            error += np.square(o-o_)
    error = np.sqrt(error/(n_classes*n_samples))
    return error

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_firing_rate(
        np.ndarray[np.float64_t, ndim=1] spike_t, float length,
        float wwidth=0.05):

    cdef int size = int(length/wwidth)
    cdef int t_
    cdef float t
    cdef np.ndarray[np.float64_t, ndim=1] firing_rate = np.zeros(size)
    cdef np.ndarray[np.float64_t, ndim=1] time = (np.arange(size)+1)*wwidth

    for t in spike_t:
        t_ = int(t/wwidth)
        if t_ < size: firing_rate[t_] += 1
    return time, firing_rate/wwidth


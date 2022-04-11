
import time

start = time.time()
import argparse
import json
import numpy as np
from brian2 import *
from lib.mySNN import mySNN
print('import: ', time.time()-start)

set_device('cpp_standalone', build_on_run=False)

parser = argparse.ArgumentParser()
parser.add_argument('config',
    type=str, help='config file (json)')
parser.add_argument('states',
    type=str, help='state file (npz)')
parser.add_argument('-s', '--stm_eles',
    type=int, nargs='+',
    default=range(20), help='stimulation electrodes')
parser.add_argument('-o', '--output',
    type=str, default=None, help='basename of output file')
parser.add_argument('--suffix',
    type=str, default='', help='suffix of output file name')
parser.add_argument('-l', '--length',
    type=float, default=1.5, help='runtime length (second)')
parser.add_argument('--stimulation_time',
    type=float, default=10., help='stimulation time (second)')
parser.add_argument('-t', '--trials',
    type=int, default=1, help='number of trials')
parser.add_argument('--seed',
    type=int, default=0, help='rundom seed')
parser.add_argument('--parallel',
    action='store_true', help='parallel')
args = parser.parse_args()

c_file = args.config
s_file = args.states
o_file = args.output
suffix = args.suffix
stm_eles = args.stm_eles
length = args.length
stimulation_time = args.stimulation_time
trials = args.trials
seed_number = args.seed
parallel = args.parallel

directory = None if parallel else 'output'

runtime = (length+stimulation_time)*second

params = None
with open(c_file, 'r') as f:
    params = json.load(f)

start = time.time()
# Network Initialization
snn = mySNN(init_weights=False, record=True, **params)
states = np.load(s_file)
print('initialization: ', time.time()-start)

# Set Spikes
n_electrodes = len(stm_eles)
input_i_ = stm_eles
input_t_ = [stimulation_time]*n_electrodes
input_i = []
input_t = []
for trial in range(trials):
    input_i.extend(stm_eles)
    input_t.extend([stimulation_time+(runtime/second)*trial]*n_electrodes)
input_t = input_t*second
snn.set_spikes(input_i, input_t)
#print(input_i)
#print(input_t)

start = time.time()
# Run
for trial in range(trials):
    seed_number_ = seed_number+trial
    seed(seed_number_)

    snn.initialize_with(states, without_connect=True if trial > 0 else False)
    snn.run_without_output(runtime)
device.build(directory=directory, compile=True, run=True, debug=False)

spike_i, spike_t = snn.get_spikes()
for trial in range(trials):
    seed_number_ = seed_number+trial
    indices = np.where((spike_t>=runtime*trial) & (spike_t<runtime*(trial+1)))[0]
    
    spike_i_ = spike_i[indices]
    spike_t_ = spike_t[indices]-runtime*trial
    if o_file:
        data = {'spiketrain': (spike_i_, (spike_t_/second).astype(np.float32)),
                'input': (input_i_, (input_t_/second).astype(np.float32)),
                'N': params['N'],
                'length': runtime/second,
                'config': c_file,
                'states': s_file,
                'stm_eles': stm_eles,
                'seed': seed_number_}
        np.savez(o_file+'_%03d%s.npz'%(trial, suffix), **data)

if parallel: device.delete()
print('run: ', time.time()-start)


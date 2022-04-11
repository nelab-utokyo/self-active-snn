
import time

start = time.time()
import argparse
import json
import numpy as np
from brian2 import *
from lib.mySNN import mySNN
print('import: ', time.time()-start)

parser = argparse.ArgumentParser()
parser.add_argument('config',
    type=str, help='config file (json)')
parser.add_argument('states',
    type=str, help='state file (npz)')
parser.add_argument('-s', '--stm_eles',
    type=int, nargs='+',
    default=range(20), help='stimulation electrodes')
parser.add_argument('-o', '--output',
    type=str, default=None, help='outpu file name (npz)')
parser.add_argument('-l', '--length',
    type=float, default=0.5, help='runtime length (second)')
parser.add_argument('--stimulation_time',
    type=float, default=10., help='stimulation time (second)')
parser.add_argument('--seed',
    type=int, default=0, help='rundom seed')
parser.add_argument('--parallel',
    action='store_true', help='parallel')
args = parser.parse_args()

c_file = args.config
s_file = args.states
o_file = args.output
stm_eles = args.stm_eles
length = args.length
stimulation_time = args.stimulation_time
seed_number = args.seed
parallel = args.parallel

start = time.time()
directory = None if parallel else 'output'
set_device('cpp_standalone', directory=directory)
seed(seed_number)
print('set device: ', time.time()-start)

runtime = (length+stimulation_time)*second

params = None
with open(c_file, 'r') as f:
    params = json.load(f)

start = time.time()
# Network Initialization
snn = mySNN(init_weights=False, record=True, **params)
snn.initialize_with(np.load(s_file))
print('initialization: ', time.time()-start)

# Set Spikes
n_electrodes = len(stm_eles)
input_i = stm_eles
input_t = ([stimulation_time]*n_electrodes)*second
snn.set_spikes(input_i, input_t)

start = time.time()
# Run
spike_i, spike_t = snn.run(runtime)
print('run: ', time.time()-start)

if o_file:
    data = {'spiketrain': (spike_i, (spike_t/second).astype(np.float32)),
            'input': (input_i, (input_t/second).astype(np.float32)),
            'N': params['N'],
            'length': runtime/second,
            'config': c_file,
            'states': s_file,
            'stm_eles': stm_eles,
            'seed': seed_number}
    np.savez(o_file, **data)
if parallel: device.delete()


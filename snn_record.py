
import argparse
import json
import numpy as np
from brian2 import *
from lib.mySNN import mySNN

parser = argparse.ArgumentParser()
parser.add_argument('config',
    type=str, help='config file (json)')
parser.add_argument('states',
    type=str, help='state file (npz)')
parser.add_argument('outfile',
    type=str, help='file name (npz)')
parser.add_argument('-l', '--length',
    type=int, default=2, help='length (second)')
parser.add_argument('--sp',
    type=float, default=0.001, help='sampling period (second)')
parser.add_argument('--spike_only',
    action='store_true', help='record only spikes')
parser.add_argument('--parallel',
    action='store_true', help='parallel')
parser.add_argument('--de',
    type=float, default=None, help='delta estdp')
parser.add_argument('--di',
    type=float, default=None, help='delta istdp')
parser.add_argument('-o', '--o_file',
    type=str, default=None, help='state file (npz)')
parser.add_argument('--seed',
    type=int, default=0, help='random seed')
args = parser.parse_args()

c_file = args.config
s_file = args.states
outfile = args.outfile
length = args.length
sp = args.sp
spike_only = args.spike_only
parallel = args.parallel
delta_estdp = args.de
delta_istdp = args.di
o_file = args.o_file
seed_number = args.seed

# Simulation setting
if spike_only:
    directory = None if parallel else 'output'
    set_device('cpp_standalone', directory=directory)
    seed(seed_number)
else:
    np.random.seed(seed_number)

runtime = length*second

# SNN initialization
params = None
with open(c_file, 'r') as f:
    params = json.load(f)
if delta_estdp is not None: params['delta_estdp'] = delta_estdp
if delta_istdp is not None: params['delta_istdp'] = delta_istdp

states = np.load(s_file)

snn = mySNN(init_weights=False,
            record=True,
            record_state=(not spike_only), record_synapse=(not spike_only),
            **params)
snn.initialize_with(states)

# Run SNN
spike_i, spike_t = snn.run(runtime)

data = {'spiketimes': (spike_i, (spike_t/second).astype(np.float32)), 'length': length}
if not spike_only:
    interval = int(sp/(defaultclock.dt/second))

    statemon = snn.get_state_monitor()
    synapse_monitors = snn.get_synapse_monitors()
    state_t = statemon.t
    state_v = statemon.v
    gtot_exc = statemon.gtot_exc
    gtot_inh = statemon.gtot_inh

    data['sp'] = sp
    data['time'] = (state_t/second).astype(np.float32)[::interval]
    data['mua'] = (state_v/mV).astype(np.float32)[:, ::interval]
    data['gtot_exc'] = gtot_exc.astype(np.float32)[:, ::interval]
    data['gtot_inh'] = gtot_inh.astype(np.float32)[:, ::interval]

    for direction in ('EE', 'EI', 'IE', 'II'):
        data['trace_g_'+direction] = synapse_monitors[direction].g.astype(np.float32)[:, ::interval]
        data['trace_x_'+direction] = synapse_monitors[direction].x.astype(np.float32)[:, ::interval]
        data['trace_w_'+direction] = synapse_monitors[direction].w.astype(np.float32)[:, ::interval]

np.savez(outfile, **data)
if o_file is not None: snn.save_states(o_file)

if parallel: device.delete()


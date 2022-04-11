
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
parser.add_argument('output',
    type=str, help='output file name (npz)')
parser.add_argument('-i', '--interval',
    type=float, default=1., help='interval (second)')
parser.add_argument('--rep',
    type=int, default=3600, help='number of repetition')
parser.add_argument('--de',
    type=float, default=None, help='delta estdp')
parser.add_argument('--di',
    type=float, default=None, help='delta istdp')
parser.add_argument('--seed',
    type=int, default=0, help='rundom seed')
parser.add_argument('--parallel',
    action='store_true', help='parallel')
parser.add_argument('--pattern_file',
    type=str, default='patterns.txt', help='pattern file (txt)')
parser.add_argument('--train_ids',
    type=int, nargs='+', default=[], help='train ids')
args = parser.parse_args()

c_file = args.config
s_file = args.states
o_file = args.output
interval = args.interval
rep = args.rep
delta_estdp = args.de
delta_istdp = args.di
seed_number = args.seed
parallel = args.parallel
pattern_file = args.pattern_file
train_ids = args.train_ids

patterns = []
with open(pattern_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        ids = line.split(' ')
        patterns.append([ int(i) for i in ids ])

if len(train_ids) == 0:
    train_ids = range(len(patterns))

pattern_ids = []
for i in train_ids:
    pattern_ids.extend([i]*int(rep/4))
np.random.shuffle(pattern_ids)
print(len(pattern_ids))

directory = None if parallel else 'output'
set_device('cpp_standalone', directory=directory)
seed(seed_number)

runtime = (rep+1)*interval*second

params = None
with open(c_file, 'r') as f:
    params = json.load(f)
if delta_estdp is not None:
    params['delta_estdp'] = delta_estdp
if delta_istdp is not None:
    params['delta_istdp'] = delta_istdp
print(params)

# Network Initialization
snn = mySNN(init_weights=False, record=False, **params)
snn.initialize_with(np.load(s_file))

# Set Spikes
input_i = []
input_t = []
for i, pattern_id in enumerate(pattern_ids):
    pattern = patterns[pattern_id]
    input_i.extend(pattern)
    input_t.extend([i*interval]*len(pattern))
input_t = input_t*second
snn.set_spikes(input_i, input_t)

# Run
_, _ = snn.run(runtime)

snn.save_states(o_file)


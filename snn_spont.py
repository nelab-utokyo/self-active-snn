
import argparse
import json
import numpy as np
from brian2 import *
from lib.mySNN import mySNN

parser = argparse.ArgumentParser()
parser.add_argument('config',
    type=str, help='config file (json)')
parser.add_argument('output',
    type=str, help='output file name (npz)')
parser.add_argument('-s', '--states',
    type=str, default=None, help='state file (npz)')
parser.add_argument('-l', '--length',
    type=int, default=0, help='runtime length (hour)')
parser.add_argument('--seed',
    type=int, default=0, help='random seed')
parser.add_argument('--parallel',
    action='store_true', help='parallel')
args = parser.parse_args()

c_file = args.config
o_file = args.output
s_file = args.states
length = args.length
seed_number = args.seed
parallel = args.parallel

# Simulation setting
directory = None if parallel else 'output'
set_device('cpp_standalone', directory=directory)
seed(seed_number)

runtime = 3600*length*second

# Generate input spike trains
params = None
with open(c_file, 'r') as f:
    params = json.load(f)

snn = mySNN(init_weights=(s_file is None), record=False, **params)
if s_file is not None:
    snn.initialize_with(np.load(s_file))

_, _ = snn.run(runtime)

snn.save_states(o_file)

if parallel: device.delete()


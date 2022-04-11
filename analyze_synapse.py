
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('infiles',
    type=str, nargs='+', help='state files (npz)')
parser.add_argument('-o', '--outfile',
    type=str, default=None, help='output file name (npz)')
args = parser.parse_args()

infiles = args.infiles
outfile = args.outfile

directions = ['EE', 'EI', 'IE', 'II']

data = {'files': []}
for direction in directions:
    data[direction] = []

for infile in infiles:
    params = np.load(infile, 'r')
    print(infile)
    data['files'].append(infile)

    for i, direction in enumerate(directions):
        W = params['w_'+direction]
        targets, sources = np.where(np.isnan(W)==False)

        strengths = W[targets, sources]
        #indices = np.where(strengths>=0.3)[0]
        #val = len(indices)

        #val = np.sum(strengths)

        val = np.mean(strengths)

        data[direction].append(val)

np.savez(outfile, **data)


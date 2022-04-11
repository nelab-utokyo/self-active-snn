
import argparse
import numpy as np
from brian2 import second
from lib.avalanche import get_avalanche_sizes, get_counts, fit_powerlaw, fit_expon
from lib.utils import linear_regression
from lib.utils_c import calc_firing_rate

parser = argparse.ArgumentParser()
parser.add_argument('infiles',
    type=str, nargs='+', help='record files (npz)')
parser.add_argument('-o', '--outfile',
    type=str, default=None, help='output file name (npz)')
parser.add_argument('-n', '--n_neurons',
    type=int, default=None, help='number of neurons (int)')
args = parser.parse_args()

infiles = args.infiles
outfile = args.outfile
n_neurons = args.n_neurons

N = None
if n_neurons is not None:
    N = n_neurons

Ds = []
BIs = []
LLRs = []
deltas = []
for infile in infiles:
    params = np.load(infile, 'r')
    if n_neurons is None:
        N = int(infile.split('/')[-1].split('.')[0].split('_')[2][1:])
    print(infile, N)

    spike_i, spike_t, length = None, None, None
    with np.load(infile, 'r') as data:
        spike_t, spike_t = data['spiketimes']
        length = data['length']
    _, firing_rate = calc_firing_rate(spike_t, length, wwidth=1.0)
    frs = sorted(firing_rate.tolist())

    # calculate D
    avalanche_sizes = get_avalanche_sizes(spike_t, unit=second, sort=True)
    sizes, counts = get_counts(avalanche_sizes)
    Ds.append(np.max(np.diff(sizes)))

    # calculate BI
    total = np.sum(frs)
    n_bins = len(frs)
    i_3rd = int(n_bins*0.85)

    f_15 = np.sum(frs[i_3rd:])/total
    BI = (f_15-0.15)/0.85

    BIs.append(BI)

    # calculate LLR
    lam, c_expon, LL_lambda = fit_expon(avalanche_sizes, 1, N)
    alpha, c_power, LL_alpha = fit_powerlaw(avalanche_sizes, 1, N)

    LLRs.append(LL_alpha-LL_lambda)

    # calculating delta q
    indices = np.where((avalanche_sizes>=1)&(avalanche_sizes<=N))[0]
    avalanche_sizes = avalanche_sizes[indices]
    sizes, counts = get_counts(avalanche_sizes)

    sizes_log10, counts_log10 = np.log10(sizes).reshape((-1, 1)), np.log10(counts).reshape((-1, 1))
    x_lreg = np.log10(range(1, N+1)).reshape((-1, 1))
    y_emp, y_lreg, range_best = linear_regression(sizes_log10, counts_log10, x_lreg)

    p_emp = np.zeros((N, 1))
    total = np.sum(counts).astype(np.float32)
    for size, count in zip(sizes, counts):
        p_emp[size-1, 0] = float(count)/total
    p_the = np.power(10, y_lreg)
    p_the = p_the/total
    #delta_p = np.sum(p_emp[1:]-p_the[1:])

    #diff = p_emp[1:, 0]-p_the[1:, 0]
    diff = p_emp[range_best[1]:, 0]-p_the[range_best[1]:, 0]
    indices = np.where(diff>0)[0]
    upper_area = np.sum(diff[indices])
    indices = np.where(diff<0)[0]
    lower_area = np.sum(-diff[indices])

    delta = upper_area if upper_area > lower_area else -lower_area

    deltas.append(delta)

data = {'Ds': Ds, 'BIs': BIs, 'LLRs': LLRs, 'deltas': deltas, 'files': infiles}
np.savez(outfile, **data)


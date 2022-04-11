
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from brian2 import second
from lib.utils_c import calc_firing_rate

parser = argparse.ArgumentParser()
parser.add_argument('records',
    type=str, nargs='+', help='record files (npz)')
parser.add_argument('-o', '--outfile',
    type=str, default=None, help='output file name (pdf)')
parser.add_argument('-b', '--before',
    type=float, default=0.2, help='time before stimulus (float)')
parser.add_argument('-a', '--after',
    type=float, default=0.5, help='time after stimulus (float)')
parser.add_argument('-p', '--plot',
    action='store_true', help='plot if stored')
args = parser.parse_args()

r_files = args.records
outfile = args.outfile
before = args.before
after = args.after
plot = args.plot

spiketrains = []
firing_rates = []
times = []
stim_end = None
for r_file in r_files:
    with np.load(r_file, 'r') as data:
        spike_i, spike_t = data['spiketrain']
        input_i, input_t = data['input']
        length = data['length']

        stim_end = np.max(input_t)
        time, firing_rate = calc_firing_rate(spike_t, length, wwidth=0.05)

        spiketrains.append((spike_i, spike_t-stim_end))
        firing_rates.append(firing_rate.tolist())
        times.append((time-stim_end).tolist())
firing_rates = np.array(firing_rates)
times = np.array(times)

# Plot
if len(spiketrains) == 1:
    fig, axs = plt.subplots(nrows=2, figsize=(12, 7), sharex=True)
    axs[0].plot(times[0], firing_rates[0], c='k')
    axs[1].plot(spiketrains[0][1], spiketrains[0][0], '.k', ms=1)
else:
    psth = np.mean(firing_rates, axis=0)
    fig, axs = plt.subplots(nrows=2, figsize=(12, 7))
    for firing_rate in firing_rates:
        axs[0].plot(times[0], firing_rate, c='k', alpha=0.2)
    axs[1].plot(times[0], psth, c='k')
axs[1].set_xlim(-before, after)
axs[1].set_ylabel('Neuron')
axs[1].set_xlabel('Time after stimulus, second')

if outfile:
    pdf = PdfPages(outfile)
    fignums = plt.get_fignums()
    for fignum in fignums:
        plt.figure(fignum)
        pdf.savefig()
    pdf.close()
if plot:
    plt.show()


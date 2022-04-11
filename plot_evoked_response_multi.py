
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from brian2 import second
from lib.utils_c import calc_firing_rate

plt.rcParams.update({
    "font.size": 8, 
    "xtick.labelsize": 7,
    "ytick.labelsize": 7
    }) 
legend_size = 7
figsize_fc = (7.48, 2)

parser = argparse.ArgumentParser()
parser.add_argument('records',
    type=str, nargs='+', help='record files (npz)')
parser.add_argument('-o', '--outfile',
    type=str, default=None, help='output file name (pdf)')
parser.add_argument('-b', '--before',
    type=float, default=0.04, help='time before stimulus (float)')
parser.add_argument('-a', '--after',
    type=float, default=0.2, help='time after stimulus (float)')
parser.add_argument('-p', '--plot',
    action='store_true', help='plot if stored')
args = parser.parse_args()

r_files = args.records
outfile = args.outfile
before = args.before
after = args.after
plot = args.plot

spiketrains = []
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
        times.append((time-stim_end).tolist())
times = np.array(times)

# Plot
fig, ax = plt.subplots(figsize=figsize_fc)
for trial, (spike_i, spike_t) in enumerate(spiketrains):
    #axs[0].plot(spiketrain[1], spiketrain[0], '.', alpha=0.5, ms=1)
    ax.plot(spike_t, [trial]*len(spike_t), '.k', alpha=0.5, ms=1)
ax.set_xlim(-before, after)
ax.set_xticks(np.arange(-before, after, 0.02))
ax.set_yticks(np.linspace(0, len(spiketrains), 5))
ax.set_ylabel('Trial')
ax.set_xlabel('Time after stimulus, second')

if outfile:
    pdf = PdfPages(outfile)
    fignums = plt.get_fignums()
    for fignum in fignums:
        plt.figure(fignum)
        pdf.savefig()
    pdf.close()
if plot:
    plt.show()


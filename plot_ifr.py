
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from brian2 import second
from lib.utils_c import calc_firing_rate

parser = argparse.ArgumentParser()
parser.add_argument('infiles',
    type=str, nargs='+', help='input files (npz)')
parser.add_argument('-o', '--output',
    type=str, default=None, help='output file name (pdf)')
parser.add_argument('-p', '--plot',
    action='store_true', help='plot if stored')
parser.add_argument('--times',
    type=str, nargs='+',
    default=['before', 'after00', 'after01', 'after02', 'after03'])
args = parser.parse_args()

infiles = args.infiles
o_file = args.output
plot = args.plot
times = np.array(args.times)

dataset = []
bin_edges = None
flag = False
for time in times:
    for infile in infiles:
        flag = False
        candidates = infile.split('/')[-1].split('.')[0].split('_')
        for candidate in candidates:
            if candidate == time:
                flag = True
                break
        if flag == False: continue

        with np.load(infile, 'r') as d:
            X = d['X']
            y = d['y']
            #stim_end = d['stim_end']
            #bin_edges = d['bin_edges']-stim_end
            bin_edges = d['bin_edges']
            wshift = d['wshift']

            dataset.append((X, y))

# Plot
if len(dataset) == 1:
    #gradient = np.linspace(0.0, 1, len(set(y)))
    #cmap = plt.get_cmap('jet')
    #for X, y in dataset:
    #    fig, axs = plt.subplots(nrows=2, figsize=(12, 5), sharex=True)
    #    for ifr, i in zip(X, y):
    #        c = cmap(gradient[i])
    #        axs[0].plot(bin_edges, np.sum(ifr, axis=0), c=c, alpha=0.2)
    #    axs[1].plot(bin_edges, np.mean(np.sum(X, axis=1), axis=0), c='k')
    #    axs[1].set_xlim(bin_edges[0], bin_edges[-1])

    fig, axs = plt.subplots(nrows=len(set(y)), figsize=(7, 5), sharex=True, sharey=True)
    for X, y in dataset:
        for ifr, i in zip(X, y):
            axs[i].plot(bin_edges, np.sum(ifr, axis=0), c='k', alpha=0.2)
    axs[-1].set_xlim(bin_edges[0], bin_edges[-1])

    fig, ax = plt.subplots(figsize=(7, 3))
    gradient = np.linspace(0.0, 1, len(set(y)))
    cmap = plt.get_cmap('jet')
    labels = sorted(list(set(y)))
    for label in labels:
        ifrs = X[np.where(y==label)[0]]
        mean_ifrs = np.mean(ifrs, axis=1)
        c = cmap(gradient[label])
        ax.plot(bin_edges, np.median(mean_ifrs, axis=0), c=c, label=label)
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.legend()
else:
    gradient = np.linspace(0.2, 1, len(times))
    cmap = plt.get_cmap('jet')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 3))
    for i, (X, y) in enumerate(dataset):
        ax.plot(bin_edges, np.median(np.sum(X, axis=1), axis=0),
            c=cmap(gradient[i]), lw=0.8, label=times[i])
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.set_xlabel('Time, second')
    ax.set_ylabel('Spike Counts, counts/second')
    ax.legend()
plt.tight_layout()

    #psth = np.mean(firing_rates, axis=0)
    #fig, axs = plt.subplots(nrows=2, figsize=(12, 7))
    #for firing_rate in firing_rates:
    #    axs[0].plot(times[0], firing_rate, c='k', alpha=0.2)
    #axs[1].plot(times[0], psth, c='k')
    #axs[1].set_xlim(-stim_end, length-stim_end)

if o_file:
    pdf = PdfPages(outfile)
    fignums = plt.get_fignums()
    for fignum in fignums:
        plt.figure(fignum)
        pdf.savefig()
    pdf.close()
if plot:
    plt.show()


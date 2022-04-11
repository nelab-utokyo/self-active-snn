
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import sem, f, friedmanchisquare, wilcoxon

plt.rcParams.update({
    "font.size": 8, 
    "xtick.labelsize": 7,
    "ytick.labelsize": 7
    }) 
legend_size = 7
figsize_fc = (7.48, 2)
figsize_sc = (3.54, 2.0)

p_threshold = 0.001
dict_ABB = { 'crt': 'critical', 'sub': 'subcritical', 'sup': 'supercritical' }

parser = argparse.ArgumentParser()
parser.add_argument('infiles',
    type=str, nargs='+', help='infiles (npz)')
parser.add_argument('-o', '--outfile',
    type=str, default=None, help='outfilename (pdf)')
parser.add_argument('-l', '--length',
    type=float, default=0.2, help='length (float)')
parser.add_argument('-p', '--plot',
    action='store_true', help='plot')
parser.add_argument('-d', '--detail',
    action='store_true', help='detail')
parser.add_argument('--conditions',
    type=str, nargs='+',
    default=None, help='conditions')
args = parser.parse_args()

infiles = args.infiles
outfile = args.outfile
length = args.length
plot = args.plot
detail = args.detail
conditions = args.conditions

labels = [ dict_ABB[cond] for cond in conditions ]

dataset = {}
for infile in infiles:
    #condition = infile.split('/')[0]
    condition = infile.split('/')[0].split('_')[-1]

    if condition in dataset.keys(): pass
    else:
        dataset[condition] = {'ifrs': [], 'coefs': []}

    print(infile)
    with np.load(infile, 'r') as data:
        ifrs = data['X']
        bin_edges = data['bin_edges']

        indices_bin = np.where(bin_edges<=length*1.1)[0]
        dataset[condition]['ifrs'].append(ifrs[:, :, indices_bin].tolist()) # trial, channel, bin
        dataset[condition]['bin_edges'] = bin_edges[indices_bin]
if conditions is None: conditions = list(dataset.keys())

bin_edges = dataset[conditions[0]]['bin_edges']

gradient = np.linspace(0.0, 0.8, len(conditions))
cmap = plt.get_cmap('viridis')
#cmap = plt.get_cmap('jet')

fig, axs = plt.subplots(nrows=2, figsize=figsize_sc, sharex=True)
data = []
for i, condition in enumerate(conditions):
    ifrs = np.array(dataset[condition]['ifrs']) # file, trial, channel, bin
    #mean_ifrs = np.mean(np.mean(ifrs, axis=2), axis=(0, 1))
    mean_ifrs = np.mean(ifrs, axis=(1, 2))

    axs[0].plot(bin_edges, np.median(mean_ifrs, axis=0),
        c=cmap(gradient[i]), alpha=1., label=dict_ABB[condition], lw=0.8)

    #data.append(np.mean(ifrs, axis=(1, 2)).tolist())
    data.append(mean_ifrs.tolist())
data = np.array(data) # condition, file, bin
data = data.T

p_values = []
for data_each in data:
    #p_value = welch_anova(data_each.T)[3]
    p_value = friedmanchisquare(*data_each.T)[1]
    p_values.append(p_value)
#p_values = np.array(p_values)*len(data)
p_values = np.array(p_values)
axs[0].set_ylabel('Firing Rate, Hz')
#axs[0].set_yscale('log')
axs[0].legend(fontsize=legend_size, ncol=1)
axs[1].plot(bin_edges, p_values, c='k', lw=1)
axs[1].axhline(p_threshold, c='k', ls='--', lw=0.2)
axs[1].set_xticks(np.arange(-0.04, length+0.04, 0.04))
axs[0].xaxis.set_minor_locator(MultipleLocator(0.02))
axs[1].xaxis.set_minor_locator(MultipleLocator(0.02))
axs[1].set_xlim(-0.04, length)
axs[1].set_xlabel('Time, second')
axs[1].set_ylabel('p-value')
axs[1].set_yscale('log')
fig.align_ylabels(axs)
plt.tight_layout()

# z-score
fig, axs = plt.subplots(nrows=2, figsize=figsize_sc, sharex=True)
data = []
for i, condition in enumerate(conditions):
    ifrs = np.array(dataset[condition]['ifrs']) # file, trial, channel, bin
    mean_ifrs = np.mean(ifrs, axis=2)

    indices = np.where(bin_edges<0)[0]
    baseline = np.mean(mean_ifrs[:, :, indices], axis=(1, 2)).reshape((-1, 1, 1))
    std = np.std(np.mean(mean_ifrs[:, :, indices], axis=2), axis=1).reshape((-1, 1, 1))

    z_traces = (mean_ifrs-baseline)/std
    mean_z_traces = np.mean(z_traces, axis=1)
    axs[0].plot(bin_edges, np.median(mean_z_traces, axis=0),
        c=cmap(gradient[i]), lw=0.8, label=dict_ABB[condition])

    data.append(mean_z_traces.tolist())
data = np.array(data) # condition, file, bin
data = data.T

p_values = []
for data_each in data:
    #p_value = welch_anova(data_each.T)[3]
    p_value = friedmanchisquare(*data_each.T)[1]
    p_values.append(p_value)
#p_values = np.array(p_values)*len(data)
p_values = np.array(p_values)
axs[0].set_ylabel('Z-score')
#axs[0].set_yscale('log')
axs[0].legend(fontsize=legend_size, ncol=1)
axs[1].plot(bin_edges, p_values, c='k', lw=1)
axs[1].axhline(p_threshold, c='k', ls='--', lw=0.2)
axs[1].set_xticks(np.arange(-0.04, length+0.04, 0.04))
axs[0].xaxis.set_minor_locator(MultipleLocator(0.02))
axs[1].xaxis.set_minor_locator(MultipleLocator(0.02))
axs[1].set_xlim(-0.04, length)
axs[1].set_xlabel('Time, second')
axs[1].set_ylabel('p-value')
axs[1].set_yscale('log')
fig.align_ylabels(axs)
plt.tight_layout()

indices_bin = np.where(p_values<p_threshold)[0]
print(indices_bin)
print(bin_edges[indices_bin])
results = []
for data_each in data.T:
    ifrs = data_each[:, indices_bin]
    mean_ifrs = np.mean(ifrs, axis=0).tolist()
    results.append(mean_ifrs)
results = np.array(results)

#stat, p_value = friedmanchisquare(*results.tolist())
#print(p_value)

fig, ax = plt.subplots(figsize=(4, 2))
for i, data_each in enumerate(results, 1):
    jitter = 0.1*np.random.random(len(data_each))-0.05
    ax.plot(jitter+i, data_each, '.', c='k', alpha=0.2)
ax.boxplot(results.tolist(), labels=labels, whis=[0, 100], widths=0.8)
ax.set_ylabel('Z-score')
plt.tight_layout()

fig, ax = plt.subplots(figsize=(7, 5))
n_significant = [ 0 for _ in range(len(conditions)) ]
for i, data_i in enumerate(results):
    for j, data_j in enumerate(results[i+1:], i+1):
        stat, p_value = wilcoxon(data_i, data_j)
        if p_value < p_threshold:
            n_significant[i] += 1
            n_significant[j] += 1

        #p_value = p_value*(len(results)-1)
        print(i, j, p_value)
        ax.text(i, j, str(p_value), size=12)
ax.set_xticks(range(len(conditions)))
ax.set_xticklabels(labels)
ax.set_yticks(range(len(conditions)))
ax.set_yticklabels(labels)
plt.tight_layout()

fig, ax = plt.subplots(figsize=(4, 2))
vmin, vmax = 1, 0
for i, data_each in enumerate(results, 1):
    jitter = 0.1*np.random.random(len(data_each))-0.05
    ax.plot(jitter+i, data_each, '.', c='k', alpha=0.2)
    vmin = min(vmin, np.min(data_each))
    vmax = max(vmax, np.max(data_each))
ax.boxplot(results.tolist(), labels=labels, whis=[0, 100], widths=0.8)
ax.set_ylim(top=vmax+(vmax-vmin)*0.3*np.max(n_significant))
ax.set_ylabel('Z-score')
plt.tight_layout()

if outfile:
    pdf = PdfPages(outfile)
    fignums = plt.get_fignums()
    for fignum in fignums:
        plt.figure(fignum)
        pdf.savefig()
    pdf.close()
if plot: plt.show()


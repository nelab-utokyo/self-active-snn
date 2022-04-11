
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import friedmanchisquare, wilcoxon

plt.rcParams.update({
    #"font.family": "Nimbus Sans",
    "font.size": 8, 
    "xtick.labelsize": 7,
    "ytick.labelsize": 7
    }) 
legend_size = 4
figsize_sc = (3.54, 2.0)

p_threshold = 0.001

parser = argparse.ArgumentParser()
parser.add_argument('infiles',
    type=str, nargs='+', help='input files (npz)')
parser.add_argument('-o', '--outfile',
    type=str, default=None, help='output file name (pdf)')
parser.add_argument('-l', '--length',
    type=float, default=0.2, help='length (float)')
parser.add_argument('-w', '--wwidth',
    type=float, default=0.02, help='wwidth (float)')
parser.add_argument('-p', '--plot',
    action='store_true', help='plot if stored')
parser.add_argument('--times',
    type=int, default=None)
parser.add_argument('--padding',
    type=int, default=1)
args = parser.parse_args()

infiles = args.infiles
o_file = args.outfile
length = args.length
wwidth = args.wwidth
plot = args.plot

times_ = None
if args.times is not None:
    times_ = ['before']
    #for t in range(args.times+1):
    for t in range(0, args.times+1, args.padding):
        times_.append('%d'%t)

times = []
dataset = {}
for infile in infiles:
    print(infile)
    t = infile.split('/')[-1].split('.')[0].split('_')[-1]
    if len(t.split('after')) > 1:
        t = t.split('after')[-1]
    if t != 'before': t = str(int(t))
    s = infile.split('/')[-2]
    if times_ is None: pass
    elif t in times_: pass
    else: continue

    if t in dataset.keys(): pass
    else:
        dataset[t] = {'ifrs': []}
        times.append(t)

    with np.load(infile, 'r') as d:
        ifrs = d['X']
        bin_edges = np.around(d['bin_edges'], 4)

        indices_bin = np.where(bin_edges<=length*1.1)[0]
        dataset[t]['ifrs'].append(ifrs[:, :, indices_bin].tolist())
        dataset[t]['bin_edges'] = bin_edges[indices_bin]
if times_ is None: pass
elif set(times_).issubset(set(times)):
    times = times_
else:
    print(times_)
    print(times)
    exit(1)

if times_ is not None: pass
elif 'before' in times:
    times.remove('before')
    times.insert(0, 'before')
print(times)

bin_edges = dataset[times[0]]['bin_edges']

# Plot
gradient = np.linspace(0.2, 1, len(times))
cmap = plt.get_cmap('jet')

fig, axs = plt.subplots(nrows=2, figsize=figsize_sc, sharex=True)
data = []
for i, time in enumerate(times):
    ifrs = np.array(dataset[time]['ifrs'])
    #mean_ifrs = np.mean(np.mean(ifrs, axis=2), axis=(0, 1))
    mean_ifrs = np.mean(np.mean(ifrs, axis=2), axis=1)
    axs[0].plot(bin_edges, np.median(mean_ifrs, axis=0),
        c=cmap(gradient[i]), lw=0.4, label=times[i])
    print(time, ifrs.shape)

    data.append(mean_ifrs.tolist())
    #std_ifrs = np.std(np.sum(ifrs, axis=2), axis=(0, 1), ddof=1)
    #ax.fill_between(bin_edges, mean_ifrs-std_ifrs, mean_ifrs+std_ifrs,
    #    color=cmap(gradient[i]), alpha=0.2)
axs[0].set_ylabel('Firing Rate, Hz')
axs[0].legend(fontsize=legend_size, ncol=3)
data = np.array(data).T
p_values = []
if len(times) > 1:
    if len(times) > 2: test = friedmanchisquare
    else: test = wilcoxon
    for data_each in data:
        #print(data_each.T.shape)
        #p_value = welch_anova(data_each.T)[3]
        p_value = test(*data_each.T.tolist())[1]
        p_values.append(p_value)
else: p_values = [1.]*len(data)
p_values = np.array(p_values)
#p_values = np.array(p_values)*len(data)
axs[1].plot(bin_edges, p_values, c='k', lw=0.4)
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
baselines = []
stds = []
for i, time in enumerate(times):
    ifrs = np.array(dataset[time]['ifrs']) # file, trial, channel, bin
    mean_ifrs = np.mean(ifrs, axis=2)

    indices = np.where(bin_edges<0)[0]
    baseline = np.mean(mean_ifrs[:, :, indices], axis=(1, 2)).reshape((-1, 1, 1))
    std = np.std(np.mean(mean_ifrs[:, :, indices], axis=2), axis=1).reshape((-1, 1, 1))

    z_traces = (mean_ifrs-baseline)/std
    mean_z_traces = np.mean(z_traces, axis=1)
    axs[0].plot(bin_edges, np.median(mean_z_traces, axis=0),
        c=cmap(gradient[i]), lw=0.4, label=times[i])
    print(time, ifrs.shape)

    data.append(mean_z_traces.tolist())
    baselines.append(baseline.reshape(-1).tolist())
    stds.append(std.reshape(-1).tolist())
    #std_ifrs = np.std(np.sum(ifrs, axis=2), axis=(0, 1), ddof=1)
    #ax.fill_between(bin_edges, mean_ifrs-std_ifrs, mean_ifrs+std_ifrs,
    #    color=cmap(gradient[i]), alpha=0.2)
axs[0].set_ylabel('Z-score')
axs[0].legend(fontsize=legend_size, ncol=3)
data = np.array(data).T
p_values = []
if len(times) > 1:
    if len(times) > 2: test = friedmanchisquare
    else: test = wilcoxon
    for data_each in data:
        #p_value = welch_anova(data_each.T)[3]
        p_value = test(*data_each.T)[1]
        p_values.append(p_value)
else: p_values = [1.]*len(data)
p_values = np.array(p_values)
axs[1].plot(bin_edges, p_values, c='k', lw=0.4)
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

fig, ax = plt.subplots(figsize=figsize_sc)
for i, baseline in enumerate(baselines, 1):
    jitter = 0.1*np.random.random(len(baseline))-0.05
    ax.plot(jitter+i, baseline, '.', c='k', alpha=0.6, ms=2)
ax.boxplot(baselines, labels=times, whis=[0, 100])
ax.set_ylabel('Baseline, Hz')
ax.set_xlabel('Time, hour' if times[0] != 'before' else 'Time after Repetitive Stimulation, hour')
plt.tight_layout()

indices_bin = np.where(np.logical_and(p_values<p_threshold, bin_edges>0))[0]
if len(indices_bin) > 0:
    indices_bin = indices_bin[np.argmin(p_values[indices_bin])]
    print(indices_bin)
    print(bin_edges[indices_bin])
    #results = []
    #for t, time in enumerate(times):
    #    ifrs = np.array(dataset[time]['ifrs'])
    #    mean_ifrs = np.mean(np.sum(ifrs, axis=2), axis=(0, 1))
    #    results.append(mean_ifrs.tolist())
    #results = np.array(results)
    results = data[indices_bin].T

    fig, ax = plt.subplots(figsize=figsize_sc)
    for i, data_each in enumerate(results, 1):
        jitter = 0.1*np.random.random(len(data_each))-0.05
        ax.plot(jitter+i, data_each, '.', c='k', alpha=0.6, ms=2)
    ax.boxplot(results.tolist(), labels=times, whis=[0, 100])
    ax.set_ylabel('Z-score')
    ax.set_xlabel('Time, hour' if times[0] != 'before' else 'Time after Repetitive Stimulation, hour')
    plt.tight_layout()

    fig, ax = plt.subplots(figsize=figsize_sc)
    data_box = []
    p_values = []
    val_max = 0
    val_min = 100000
    for i, data_each in enumerate(results[1:], 1):
        stat, p_value = wilcoxon(data_each, results[0])
        #p_value = p_value*(len(results)-1)
        print(p_value)
        diff = (data_each-results[0])/results[0]*100.
        jitter = 0.1*np.random.random(len(diff))-0.05
        ax.plot(jitter+i, diff, '.', c='k', alpha=0.6, ms=2)

        data_box.append(diff.tolist())
        p_values.append(p_value)
        if val_max < np.max(diff): val_max = np.max(diff)
        if val_min > np.min(diff): val_min = np.min(diff)
    pos_y = (val_max-val_min)*0.1+val_max
    for pos_x, p_value in enumerate(p_values, 1):
        if p_value < p_threshold:
            ax.text(pos_x, pos_y, '*', ha='center', size=10)
    ax.boxplot(data_box, labels=times[1:], whis=[0, 100])
    ax.set_ylim(top=(val_max-val_min)*0.3+val_max)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_ylabel('Rate of Increase, %')
    ax.set_xlabel('Time, hour' if times[0] != 'before' else 'Time after Repetitive Stimulation, hour')
    ax.axhline(0, color='k', ls='--', lw=0.2)
    plt.tight_layout()

    fig, ax = plt.subplots(figsize=figsize_sc)
    ax.plot(p_values, c='k')
    ax.set_yscale('log')
    #ax.set_xlabel('Time, hour')
    plt.tight_layout()

if o_file:
    pdf = PdfPages(o_file)
    fignums = plt.get_fignums()
    for fignum in fignums:
        plt.figure(fignum)
        pdf.savefig()
    pdf.close()
if plot:
    plt.show()


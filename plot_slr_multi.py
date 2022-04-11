
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import sem, f, friedmanchisquare, wilcoxon, binom_test

plt.rcParams.update({
    #"font.family": "Nimbus Sans",
    "font.size": 8, 
    "xtick.labelsize": 7,
    "ytick.labelsize": 7
    }) 
legend_size = 4
figsize_fc = (7.48, 2)
figsize_sc = (3.54, 2.0)

p_threshold = 0.001

parser = argparse.ArgumentParser()
parser.add_argument('infiles',
    type=str, nargs='+', help='infiles (npz)')
parser.add_argument('-o', '--outfile',
    type=str, default=None, help='outfilename (pdf)')
parser.add_argument('-l', '--length',
    type=float, default=0.2, help='length (float)')
parser.add_argument('-w', '--wwidth',
    type=float, default=0.02, help='wwidth (float)')
parser.add_argument('-p', '--plot',
    action='store_true', help='plot')
parser.add_argument('-d', '--detail',
    action='store_true', help='detail')
parser.add_argument('--times',
    type=int, default=None)
parser.add_argument('--padding',
    type=int, default=1)
args = parser.parse_args()

infiles = args.infiles
outfile = args.outfile
length = args.length
wwidth = args.wwidth
plot = args.plot
detail = args.detail

times_ = None
if args.times is not None:
    times_ = ['before']
    for t in range(0, args.times+1, args.padding):
        #times_.append('after%02d'%t)
        times_.append('%d'%t)

times = []
dataset = {}
for infile in infiles:
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
        dataset[t] = {'accs': [], 'coefs': []}
        times.append(t)

    print(infile)
    with np.load(infile, 'r') as data:
        #i = np.where(data['Cs']==10)[0]
        i = 0
        dataset[t]['n_splits'] = data['n_splits']

        bin_edges = np.around(data['bin_edges'], 4)
        indices_bin = np.where(bin_edges<=length*1.1)[0]
        dataset[t]['bin_edges'] = bin_edges[indices_bin]

        dataset[t]['accs'].append(data['accs'][indices_bin, i].tolist())
        #print(data['accs'][indices_bin, i].tolist())
        dataset[t]['coefs'].append(data['coefs'][indices_bin, i].tolist())
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

step = 15
bin_edges = dataset[times[0]]['bin_edges']
n_splits = dataset[times[0]]['n_splits']
n_classes = np.array(dataset[times[0]]['coefs']).shape[3]

gradient = np.linspace(0.2, 1, len(times))
cmap = plt.get_cmap('jet')

fig, axs = plt.subplots(nrows=2, figsize=figsize_sc, sharex=True)
data = []
for t, time in enumerate(times):
    accs = np.array(dataset[time]['accs'])
    #mean_accs = np.mean(accs, axis=(0, 2))
    mean_accs = np.mean(accs, axis=2)
    trace = np.median(mean_accs, axis=0)
    color = cmap(gradient[t])

    axs[0].axhline(1./n_classes, c='k', ls='--', lw=0.2)
    axs[0].plot(bin_edges, trace,
        c=color, alpha=1., label=time, lw=0.4)

    data.append(np.mean(accs, axis=2).tolist())
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
#p_values = np.array(p_values)*len(data)
p_values = np.array(p_values)
axs[0].set_ylabel('Accuracy')
#axs[0].set_yscale('log')
axs[0].legend(fontsize=legend_size, ncol=3)
indices = np.where(bin_edges>0)[0]
axs[1].plot(bin_edges[indices], p_values[indices], c='k', lw=0.4)
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

fig, ax = plt.subplots(nrows=1, figsize=figsize_sc)
for t, time in enumerate(times):
    accs = np.array(dataset[time]['accs'])
    mean_accs = np.mean(accs, axis=2)
    trace = np.median(mean_accs, axis=0)
    color = cmap(gradient[t])

    ax.axhline(1./n_classes, c='k', ls='--', lw=0.2)
    ax.plot(bin_edges, trace,
        c=color, alpha=1., label=time, lw=0.4)
ax.set_xticks(np.arange(-0.04, length+0.04, 0.04))
ax.xaxis.set_minor_locator(MultipleLocator(0.02))
ax.set_xlim(-0.04, length)
ax.set_ylim(top=0.35)
ax.set_xlabel('Time, second')
ax.set_ylabel('Accuracy')
ax.legend(fontsize=legend_size, ncol=3)
plt.tight_layout()

fig, ax = plt.subplots(figsize=figsize_sc)
for t, time in enumerate(times):
    coefs = np.array(dataset[time]['coefs'])
    color = cmap(gradient[t])

    ax.plot(bin_edges, np.mean(coefs, axis=(0, 2, 3)),
        c=color, alpha=1., label=time, lw=0.4)
ax.set_xticks(np.arange(-0.04, length+0.04, 0.04))
ax.xaxis.set_minor_locator(MultipleLocator(0.02))
ax.set_xlim(-0.04, length)
ax.set_xlabel('Time, second')
ax.set_ylabel('#Coefficients')
plt.legend(fontsize=legend_size, ncol=3)
plt.tight_layout()

#indices_bin = np.where((bin_edges>0.05)&(bin_edges<=length*1.1))[0]
indices_bin = np.where((bin_edges>0.0)&(p_values<=p_threshold))[0]
if len(indices_bin) > 0:
    indices_bin = indices_bin[np.argmin(p_values[indices_bin])]
    #indices_bin = np.where(p_values<p_threshold)[0]
    print(indices_bin)
    print(bin_edges[indices_bin])
    results = []
    for t, time in enumerate(times):
        accs = np.array(dataset[time]['accs'])
        accs = accs[:, indices_bin]
        #mean_accs = np.mean(accs, axis=(1, 2)).tolist()
        mean_accs = np.mean(accs, axis=1).tolist()
        results.append(mean_accs)
    results = np.array(results)

    #stat, p_value = friedmanchisquare(*results.tolist())
    #print(p_value)

    #times_num = [ int(time.split('after')[-1]) if time.split('after')[-1] != 'before' else 'before' for time in times ]

    fig, ax = plt.subplots(figsize=figsize_sc)
    for i, data_each in enumerate(results, 1):
        jitter = 0.1*np.random.random(len(data_each))-0.05
        ax.plot(jitter+i, data_each, '.', c='k', alpha=0.6, ms=2)
    ax.boxplot(results.tolist(), labels=times, whis=[0, 100])
    ax.grid(axis='y')
    ax.set_xlabel('Time, hour' if times[0] != 'before' else 'Time after Repetitive Stimulation, hour')
    ax.set_ylabel('Mean Accuracy')
    plt.tight_layout()

    fig, ax = plt.subplots(figsize=figsize_sc)
    data_box = []
    p_values = []
    val_max = 0
    val_min = 100000
    for i, data_each in enumerate(results[1:], 1):
        stat, p_value = wilcoxon(data_each, results[0])
        #stat, p_value = wilcoxon(np.concatenate([data_each, data_each, data_each]).tolist(),
        #                        np.concatenate([results[0], results[0], results[0]]))
        p_value = p_value*(len(results)-1)
        print(p_value)
        #diff = data_each-results[0]
        diff = (data_each-results[0])/results[0]*100.
        jitter = 0.1*np.random.random(len(diff))-0.05
        ax.plot(jitter+i, diff, '.', c='k', alpha=0.6, ms=2)

        data_box.append(diff.tolist())
        p_values.append(p_value)
        if val_max < np.max(diff): val_max = np.max(diff)
        if val_min > np.min(diff): val_min = np.min(diff)
    pos_y = (val_max-val_min)*0.1+val_max
    for pos_x, p_value in enumerate(p_values, 1):
        if p_value < p_threshold: ax.text(pos_x, pos_y, '*', ha='center', size=10)
    ax.boxplot(data_box, labels=times[1:], whis=[0, 100])
    ax.grid(axis='y')
    ax.set_ylim(top=(val_max-val_min)*0.3+val_max)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_xlabel('Time, hour' if times[0] != 'before' else 'Time after Repetitive Stimulation, hour')
    ax.set_ylabel('Rate of Increase, %')
    ax.axhline(0, color='k', ls='--', lw=0.2)
    plt.tight_layout()

    fig, ax = plt.subplots(figsize=figsize_sc)
    ax.plot(p_values, c='k')
    ax.axhline(p_threshold, c='k', ls='--', lw=0.2)
    ax.set_yscale('log')
    plt.tight_layout()

# fading memory property
decay_times = []
ind_bin = np.where(bin_edges > 0)[0]
for t, time in enumerate(times):
    accs = np.array(dataset[time]['accs'])

    tmp_samples = []
    for accs_each in accs:
        decay_time = length
        for result, bin_edge in zip(accs_each[ind_bin], bin_edges[ind_bin]):
            x = len(np.where(result==1)[0])
            n = len(result)
            p_value = binom_test(x, n, 1./n_classes, 'greater')
            if p_value > 0.05:
                decay_time = bin_edge
                break
        tmp_samples.append(decay_time)
    decay_times.append(tmp_samples)
fig, ax = plt.subplots(figsize=figsize_sc)
ax.boxplot(decay_times, labels=times, whis=[0, 100])
ax.grid(axis='y')
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
ax.set_xlabel('Time, hour' if times[0] != 'before' else 'Time after Repetitive Stimulation, hour')
ax.set_ylabel('Decay Time, second')
plt.tight_layout()

if outfile:
    pdf = PdfPages(outfile)
    fignums = plt.get_fignums()
    for fignum in fignums:
        plt.figure(fignum)
        pdf.savefig()
    pdf.close()
if plot: plt.show()


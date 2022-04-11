
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import sem

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
parser.add_argument('--times',
    type=str, nargs='+',
    default=['before', 'after00', 'after01', 'after02', 'after03'])
args = parser.parse_args()

infiles = args.infiles
outfile = args.outfile
length = args.length
plot = args.plot
detail = args.detail
times_ = args.times

times = []
dataset = {}
flag = False
for time in times_:
    for infile in infiles:
        flag = False
        candidates = infile.split('/')[-1].split('.')[0].split('_')
        for candidate in candidates:
            if candidate == time:
                flag = True
                break
        if flag == False: continue

        print(infile)
        times.append(time)
        dataset[time] = {}
        with np.load(infile, 'r') as data:
            dataset[time]['Cs'] = data['Cs']
            #dataset[time]['stim_end'] = data['stim_end']
            dataset[time]['n_splits'] = data['n_splits']

            #bin_edges = data['bin_edges']-data['stim_end']
            bin_edges = data['bin_edges']
            indices_bin = np.where(bin_edges<=length)[0]
            dataset[time]['accs'] = data['accs'][indices_bin]
            dataset[time]['coefs'] = data['coefs'][indices_bin]
            dataset[time]['bin_edges'] = bin_edges[indices_bin]

step = 15
Cs = dataset[times[0]]['Cs']
#stim_end = dataset[times[0]]['stim_end']
bin_edges = dataset[times[0]]['bin_edges']
n_splits = dataset[times[0]]['n_splits']
n_classes = dataset[times[0]]['coefs'].shape[3]

gradient = np.linspace(0.2, 1, len(times))
cmap = plt.get_cmap('jet')

fig, axs = plt.subplots(nrows=len(Cs), figsize=(7, len(Cs)+1), sharex=True)
if len(Cs) == 1: axs = [axs]
for t, time in enumerate(times):
    accs = dataset[time]['accs']
    color = cmap(gradient[t])
    for i, C in enumerate(Cs):
        axs[i].axhline(1./n_classes, c='k', ls='--', lw=0.2)
        axs[i].plot(bin_edges, np.mean(accs[:, i], axis=1),
            c=color, alpha=1., label=time)
        #for j in range(n_splits):
        #    axs[i].plot(bin_edges, accs[:, i, j],
        #        c=color, alpha=0.2, label=time if j == 0 else None)
for i, C in enumerate(Cs):
    axs[i].set_title('C=%.1f'%C)
axs[-1].set_xlim(-0.02, length)
plt.legend(fontsize='xx-small')
plt.tight_layout()

fig, axs = plt.subplots(nrows=len(Cs), figsize=(7, len(Cs)+1), sharex=True)
if len(Cs) == 1: axs = [axs]
for t, time in enumerate(times):
    coefs = dataset[time]['coefs']
    print(time, coefs.shape)
    color = cmap(gradient[t])
    for i, C in enumerate(Cs):
        axs[i].plot(bin_edges, np.mean(coefs[:, i], axis=(1, 2)),
            c=color, alpha=1., label=time)
        #for j in range(n_splits):
        #    axs[i].plot(bin_edges, np.mean(coefs[:, i, j], axis=1),
        #        c=color, alpha=0.2, label=time if j == 0 else None)
for i, C in enumerate(Cs):
    axs[i].set_title('C=%.1f'%C)
axs[-1].set_xlim(-0.02, length)
#plt.legend(fontsize='xx-small')
plt.tight_layout()

fig, axs = plt.subplots(nrows=len(Cs), figsize=(7, len(Cs)+1), sharex=True)
if len(Cs) == 1: axs = [axs]
indices_bin = np.where((bin_edges>0.0)&(bin_edges<=length))[0]
print(bin_edges[indices_bin])
X = []
for time in times:
    mean_accs = np.mean(dataset[time]['accs'][indices_bin], axis=(0, 2))
    X.append(mean_accs.tolist())
X = np.array(X)
print(X)
for i, mean_acc in enumerate(X.T):
    axs[i].plot(mean_acc, c='k')
    axs[i].set_title('C=%.1f'%Cs[i])
axs[-1].set_xticks(range(len(times)))
axs[-1].set_xticklabels(times)
plt.tight_layout()

indices_bin = np.where((bin_edges>0.0)&(bin_edges<=length))[0]
X = []
for time in times:
    mean_accs = np.mean(dataset[time]['accs'][indices_bin], axis=(0, 2))
    X.append(mean_accs.tolist())
X = np.array(X)
indices = np.argmax(X, axis=1)
data_box = []
for i, time in zip(indices, times):
    mean_accs = np.mean(dataset[time]['accs'][indices_bin][:, i], axis=0)
    data_box.append(mean_accs.tolist())
fig, ax = plt.subplots(figsize=(7, 2))
for i, data_each in enumerate(data_box, 1):
    jitter = 0.1*np.random.random(len(data_each))-0.05
    ax.plot(jitter+i, data_each, '.', c='k', alpha=0.2)
ax.boxplot(data_box, labels=times, whis=[0, 100])
plt.tight_layout()

fig, ax = plt.subplots(figsize=(7, 2))
sems = []
for data_each in data_box:
    sems.append(sem(data_each))
ax.errorbar(range(len(data_box)), np.mean(np.array(data_box), axis=1), c='k', yerr=sems)
ax.set_xticks(range(len(times)))
ax.set_xticklabels(times)
plt.tight_layout()

#if detail:
#    for time in times:
#        accs = dataset[time]['accs']
#        coefs = dataset[time]['coefs']
#        Cs = dataset[time]['Cs']
#        bin_edges = dataset[time]['bin_edges']
#        stim_end = dataset[time]['stim_end']
#        n_Cs = len(Cs)
#
#        fig, ax = plt.subplots(figsize=(9, 3))
#        im = ax.imshow(accs.T, aspect='auto')
#        ax.invert_yaxis()
#        ax.set_xticks(range(0, len(bin_edges), step))
#        ax.set_xticklabels(xticks)
#        ax.set_yticks(range(n_Cs))
#        ax.set_yticklabels(Cs)
#        plt.colorbar(im, ax=ax)
#        plt.suptitle(infile)
#        plt.tight_layout()
#
#        _, n_Cs, n_classes = coefs.shape
#        fig, axs = plt.subplots(nrows=n_classes, figsize=(9, 7), sharex=True)
#        for c in range(n_classes):
#            for i, C in enumerate(Cs):
#                axs[c].plot(coefs[:, i, c], label=C)
#        axs[-1].set_xticks(range(0, len(bin_edges), step))
#        axs[-1].set_xticklabels(xticks)
#        axs[-1].set_xlim(-stim_end, None)
#        plt.legend(fontsize='xx-small')
#        plt.tight_layout()

if outfile:
    pdf = PdfPages(outfile)
    fignums = plt.get_fignums()
    for fignum in fignums:
        plt.figure(fignum)
        pdf.savefig()
    pdf.close()
if plot: plt.show()


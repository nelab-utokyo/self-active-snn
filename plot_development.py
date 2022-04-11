
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams.update({
    "font.size": 8, 
    "xtick.labelsize": 7,
    "ytick.labelsize": 7
    }) 
legend_size = 6

parser = argparse.ArgumentParser()
parser.add_argument('infiles',
    type=str, nargs='+', help='infiles (npz)')
parser.add_argument('-i', '--indices',
    type=str, nargs='+', default=[], help='indices (npz)')
parser.add_argument('-o', '--outfile',
    type=str, default=None, help='file name (pdf)')
parser.add_argument('-p', '--plot',
    action='store_true', help='plot if stored')
parser.add_argument('--padding',
    type=int, default=12, help='padding (int)')
parser.add_argument('-z', '--zero_equal_before',
    action='store_true', help='zero equal before')
args = parser.parse_args()

infiles = args.infiles
indices = args.indices
outfile = args.outfile
plot = args.plot
padding = args.padding
zero_equal_before = args.zero_equal_before

directions = ['EE', 'EI', 'IE', 'II']
colors = {
    'EE': 'red',
    'EI': 'coral',
    'IE': 'cyan',
    'II': 'blue'}

data = { direction: [] for direction in directions }
mean_weight_trace = { direction: None for direction in directions }
mean_weight_std = { direction: None for direction in directions }
ticklabels = None
for infile in infiles:
    print(infile)
    data_each = np.load(infile)
    #print(data_each['files'])
    if zero_equal_before:
        ticklabels = []
        for fname in data_each['files']:
            blocks = fname.split('after')
            if len(blocks) > 1:
                ticklabels.append('%d'%int(blocks[-1].split('.')[0]))
            else:
                ticklabels.append('before')
    for direction in directions:
        data[direction].append(data_each[direction])
for direction in directions:
    data[direction] = np.array(data[direction])
if len(infiles) == 1:
    for direction in directions:
        mean_weight_trace[direction] = data[direction][0]
else:
    for direction in directions:
        mean_weight_std[direction] = np.std(data[direction], axis=0, ddof=1)
        mean_weight_trace[direction] = np.mean(data[direction], axis=0)
x = range(len(mean_weight_trace['EE']))

#ticklabels = None
#if zero_equal_before:
#    ticklabels = ['before']
#    for i in range(len(x)-1):
#        ticklabels.append('%d'%i)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 1.4))
for direction in directions:
    ax.plot(mean_weight_trace[direction], label=direction, c=colors[direction])
    if mean_weight_std[direction] is not None:
        y1 = mean_weight_trace[direction]-mean_weight_std[direction]
        y2 = mean_weight_trace[direction]+mean_weight_std[direction]
        ax.fill_between(x, y1, y2, color=colors[direction], alpha=0.5)
ax.set_xticks(range(0, len(x), padding))
if zero_equal_before: ax.set_xticklabels(ticklabels)
ax.set_xlabel('Time, hour' if not zero_equal_before else 'Time after Repetitive Stimulation, hour')
ax.set_ylabel('Average Weight')
ax.set_xlim(0, len(x)-1)
ax.yaxis.set_major_locator(plt.MaxNLocator(4))
#plt.legend(fontsize='x-small')
plt.tight_layout()

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 2))
for i, direction in enumerate(directions):
    ax.plot(x, np.array(x)+i, c=colors[direction], label=direction)
ax.set_xticks(range(0, len(x), padding))
if zero_equal_before: ax.set_xticklabels(ticklabels)
ax.set_xlim(0, len(x)-1)
ax.yaxis.set_major_locator(plt.MaxNLocator(3))
plt.legend(fontsize=legend_size)
plt.tight_layout()

if len(indices) > 0:
    Ds_list, BIs_list, LLRs_list, deltas_list = [], [], [], []
    D_std, BI_std, LLR_std, delta_std = None, None, None, None
    for infile in indices:
        with np.load(infile) as data:
            Ds_list.append(data['Ds'])
            BIs_list.append(data['BIs'])
            LLRs_list.append(data['LLRs'])
            deltas_list.append(data['deltas'])
    if len(indices) == 1:
        D_trace = Ds_list[0]
        BI_trace = BIs_list[0]
        LLR_trace = LLRs_list[0]
        delta_trace = deltas_list[0]
    else:
        D_std = np.std(np.array(Ds_list), axis=0, ddof=1)
        BI_std = np.std(np.array(BIs_list), axis=0, ddof=1)
        LLR_std = np.std(np.array(LLRs_list), axis=0, ddof=1)
        delta_std = np.std(np.array(deltas_list), axis=0, ddof=1)

        D_trace = np.mean(np.array(Ds_list), axis=0)
        BI_trace = np.mean(np.array(BIs_list), axis=0)
        LLR_trace = np.mean(np.array(LLRs_list), axis=0)
        delta_trace = np.mean(np.array(deltas_list), axis=0)
    x = range(len(D_trace))

    fig, axs = plt.subplots(nrows=5, figsize=(5, 5), sharex=True)
    axs[0].plot(LLR_trace, c='k')
    if LLR_std is not None:
        axs[0].fill_between(x, LLR_trace-LLR_std, LLR_trace+LLR_std, color='grey', alpha=0.5)
    axs[0].axhline(0, c='k', ls='--', lw=0.2)
    axs[0].set_ylabel('LLR')
    axs[0].yaxis.set_major_locator(plt.MaxNLocator(3))

    axs[1].plot(D_trace, c='k')
    if D_std is not None:
        axs[1].fill_between(x, D_trace-D_std, D_trace+D_std, color='grey', alpha=0.5)
    axs[1].set_ylabel('D')
    axs[1].yaxis.set_major_locator(plt.MaxNLocator(3))

    axs[2].plot(delta_trace, c='k')
    if delta_std is not None:
        axs[2].fill_between(x, delta_trace-delta_std, delta_trace+delta_std, color='grey', alpha=0.5)
    axs[2].axhline(0, c='k', ls='--', lw=0.2)
    axs[2].set_ylabel(r'$\Delta q$')
    axs[2].yaxis.set_major_locator(plt.MaxNLocator(3))

    axs[3].plot(BI_trace, c='k')
    if BI_std is not None:
        axs[3].fill_between(x, BI_trace-BI_std, BI_trace+BI_std, color='grey', alpha=0.5)
    axs[3].set_ylabel('BI')
    axs[3].yaxis.set_major_locator(plt.MaxNLocator(3))

    for direction in directions:
        axs[4].plot(mean_weight_trace[direction], label=direction, c=colors[direction])
        if mean_weight_std[direction] is not None:
            y1 = mean_weight_trace[direction]-mean_weight_std[direction]
            y2 = mean_weight_trace[direction]+mean_weight_std[direction]
            axs[4].fill_between(x, y1, y2, color=colors[direction], alpha=0.5)
    axs[4].set_xticks(range(0, len(x), padding))
    if zero_equal_before: axs[4].set_xticklabels(ticklabels)
    axs[4].set_xlabel('Time, hour' if not zero_equal_before else 'Time after Repetitive Stimulation, hour')
    axs[4].set_ylabel('Average Weight')
    axs[4].set_xlim(0, len(x)-1)
    axs[4].yaxis.set_major_locator(plt.MaxNLocator(4))
    fig.align_ylabels(axs)
    plt.tight_layout()

if outfile:
    pdf = PdfPages(outfile)
    fignums = plt.get_fignums()
    for fignum in fignums:
        plt.figure(fignum)
        pdf.savefig()
    pdf.close()
if plot: plt.show()


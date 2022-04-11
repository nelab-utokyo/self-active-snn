
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from brian2 import second
from matplotlib.backends.backend_pdf import PdfPages
from lib.avalanche import get_avalanche_sizes, get_counts, fit_powerlaw
from lib.utils import linear_regression

plt.rcParams.update({
    "font.size": 8, 
    "xtick.labelsize": 7,
    "ytick.labelsize": 7
    }) 
legend_size = 6
figsize_fc = (7.48, 4)
figsize_sc = (3.54, 2)

parser = argparse.ArgumentParser()
parser.add_argument('records',
    type=str, nargs='+', help='record files (npz)')
parser.add_argument('-o', '--outfile',
    type=str, default=None, help='file name (pdf)')
parser.add_argument('-p', '--plot',
    action='store_true', help='plot if stored')
parser.add_argument('-d', '--detail',
    action='store_true', help='plot details if stored')
parser.add_argument('-n', '--n_neurons',
    type=int, default=None, help='number of neurons (int)')
args = parser.parse_args()

r_files = args.records
outfile = args.outfile
plot = args.plot
detail = args.detail
n_neurons = args.n_neurons

N = None
if n_neurons is not None:
    N = n_neurons

#deltas = []
#deltas_new = []
upper_area = []
lower_area = []
#deltas_abs = []
params = [] if not detail else None
for r_file in r_files:
    if n_neurons is None:
        N = int(r_file.split('/')[-1].split('.')[0].split('_')[2][1:])
    print(r_file, N)

    spike_i, spike_t = None, None
    with np.load(r_file, 'r') as data:
        spike_i, spike_t = data['spiketimes']
    blocks = r_file.split('/')[-1].split('.')[0].split('_')
    if params is not None: params.append([float(blocks[6][1:])/100, float(blocks[7][1:])/100])

    avalanche_sizes = get_avalanche_sizes(spike_t, unit=second, sort=True)

    indices = np.where((avalanche_sizes>=1)&(avalanche_sizes<=N))[0]
    avalanche_sizes = avalanche_sizes[indices]
    sizes, counts = get_counts(avalanche_sizes)

    # Linear Regression
    sizes_log10, counts_log10 = np.log10(sizes).reshape((-1, 1)), np.log10(counts).reshape((-1, 1))
    x_lreg = np.log10(range(1, N+1)).reshape((-1, 1))
    y_emp, y_lreg, range_best = linear_regression(sizes_log10, counts_log10, x_lreg)
    #delta = np.sum(y_emp-y_lreg)
    #print(delta, params[-1])

    p_emp = np.zeros((N, 1))
    total = np.sum(counts).astype(np.float32)
    for size, count in zip(sizes, counts):
        p_emp[size-1, 0] = float(count)/total
    p_the = np.power(10, y_lreg)
    p_the = p_the/total
    #delta = np.sum(p_emp[1:]-p_the[1:])

    #print(p_the[1:].reshape(-1))
    #print(p_emp[1:].reshape(-1))
    #delta_abs = np.mean(np.abs(p_emp[1:]-p_the[1:]))
    #indices = np.where(p_emp == 0)[0]
    #p_emp[indices] = p_the[1:].min()
    #print(p_emp.reshape(-1))
    #delta_abs = np.abs(np.log10(p_emp[1:])-np.log10(p_the[1:])).mean()

    #deltas.append(delta)

    #y1, y2 = np.cumsum(p_emp[1:, 0]), np.cumsum(p_the[1:, 0])
    #diff = y1-y2
    #i = np.argmax(np.abs(diff))
    #deltas_new.append(diff[i])

    #diff = p_emp[1:, 0]-p_the[1:, 0]
    diff = p_emp[range_best[1]:, 0]-p_the[range_best[1]:, 0]
    indices = np.where(diff>0)[0]
    upper_area.append(np.sum(diff[indices]))
    indices = np.where(diff<0)[0]
    lower_area.append(np.sum(diff[indices]))

    #y1, y2 = np.cumsum(p_emp[1:, 0]), np.cumsum(p_the[1:, 0])
    #deltas_abs.append(np.abs(y1-y2).max())

    #deltas_abs.append(np.abs(delta))

    #if detail:
    #    #fig, ax = plt.subplots(nrows=1, figsize=(3, 2))
    #    #ax.plot(sizes_log10, counts_log10, '.k', ms=2)
    #    #ax.plot(x_lreg, y_lreg, c='r', ms=2)
    #    #ax.set_title('%f, %d'%(delta, range_best[1]))

    #    x = np.arange(1, N+1)

    #    fig, ax = plt.subplots(ncols=1, figsize=(3, 2))
    #    rates = counts.astype(np.float32)/np.sum(counts)
    #    y1, y2 = p_emp[1:, 0], p_the[1:, 0]
    #    ax.plot(sizes, rates, '.k', ms=1)
    #    ax.plot(x, p_emp, c='k', ls='-', lw=0.6)
    #    ax.plot(x, p_the, c='grey', ls='-', lw=0.8)
    #    ax.fill_between(x[1:], y1, y2, where=(y1>y2), color='r', alpha=0.4)
    #    ax.fill_between(x[1:], y1, y2, where=(y1<y2), color='b', alpha=0.4)
    #    ax.set_xlabel('Avalanche Size, []')
    #    ax.set_ylabel('Probability, []')
    #    ax.set_xscale('log')
    #    ax.set_yscale('log')
    #    ax.set_xlim(0.9, np.max(avalanche_sizes)*1.2)
    #    ax.set_ylim(np.min(rates)*0.5, np.max(rates)*2)
    #    #fig.suptitle(r'$\Delta p=%.3f$'%delta)
    #    plt.tight_layout()

    #    fig, ax = plt.subplots(ncols=1, figsize=(3, 2))
    #    y1, y2 = np.cumsum(p_emp[1:, 0]), np.cumsum(p_the[1:, 0])
    #    ax.plot(x[1:], y1, c='k', ls='-', lw=0.6)
    #    ax.plot(x[1:], y2, c='grey', ls='-', lw=0.8)
    #    ax.fill_between(x[1:], y1, y2, where=(y1>y2), color='r', alpha=0.4)
    #    ax.fill_between(x[1:], y1, y2, where=(y1<y2), color='b', alpha=0.4)
    #    ax.set_xlabel('Avalanche Size, []')
    #    ax.set_ylabel('Cumulative Probability, []')
    #    ax.set_xscale('log')
    #    #ax.set_yscale('log')
    #    ax.set_xlim(0.9, np.max(avalanche_sizes)*1.2)
    #    #ax.set_ylim(np.min(rates)*0.5, np.max(rates)*2)
    #    #fig.suptitle(r'$\Delta p=%.3f$'%delta)
    #    plt.tight_layout()

    #    fig, ax = plt.subplots(ncols=1, figsize=(3, 2))
    #    diff = y1-y2
    #    zeros = np.zeros(len(y1))
    #    ax.plot(x[1:], diff, c='k', ls='-', lw=0.6)
    #    ax.fill_between(x[1:], diff, zeros, where=(diff>zeros), color='r', alpha=0.4)
    #    ax.fill_between(x[1:], diff, zeros, where=(diff<zeros), color='b', alpha=0.4)
    #    ax.axhline(0, c='k', ls='--', lw=0.6)
    #    ax.set_xlabel('Avalanche Size, []')
    #    ax.set_ylabel('Difference, []')
    #    ax.set_xscale('log')
    #    #ax.set_yscale('log')
    #    ax.set_xlim(0.9, np.max(avalanche_sizes)*1.2)
    #    #ax.set_ylim(np.min(rates)*0.5, np.max(rates)*2)
    #    #fig.suptitle(r'$\Delta p=%.3f$'%delta)
    #    plt.tight_layout()

if not detail:
    b_list = np.array(sorted(list(set(np.array(params)[:, 0].tolist()))))
    a_list = np.array(sorted(list(set(np.array(params)[:, 1].tolist()))))
    upper_area_map = np.zeros((len(b_list), len(a_list)), dtype=np.float32)
    lower_area_map = np.zeros((len(b_list), len(a_list)), dtype=np.float32)
    #delta_map = np.zeros((len(b_list), len(a_list)), dtype=np.float32)
    #delta_new_map = np.zeros((len(b_list), len(a_list)), dtype=np.float32)
    #for delta, delta_new, (b, a) in zip(deltas, deltas_new, params):
    #    i = np.where(b_list==b)[0][0]
    #    j = np.where(a_list==a)[0][0]

    #    delta_map[i, j] = delta
    #    delta_new_map[i, j] = delta_new
    #delta_abs_map = np.abs(delta_new_map)
    for upper_area_, lower_area_, (b, a) in zip(upper_area, lower_area, params):
        i = np.where(b_list==b)[0][0]
        j = np.where(a_list==a)[0][0]

        upper_area_map[i, j] = upper_area_
        lower_area_map[i, j] = lower_area_

    #fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    #cmap = cm.RdBu_r
    #norm = cm.colors.Normalize(vmax=abs(delta_map).max(), vmin=-abs(delta_map).max())
    #im = ax.imshow(delta_map, aspect='equal', origin='lower', cmap=cmap, norm=norm)
    #ax.set_yticks(range(len(b_list)))
    #ax.set_xticks(range(len(a_list)))
    #ax.set_yticklabels(b_list)
    #ax.set_xticklabels(a_list)
    #ax.set_xlabel(r'$\beta_\mathrm{I}$')
    #ax.set_ylabel(r'$\beta_\mathrm{E}$')
    #for i in range(len(b_list)):
    #    for j in range(len(a_list)):
    #        text = ax.text(j, i, "%.3f"%delta_map[i, j],
    #                       ha="center", va="center", color="k", size=6)
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="4%", pad=0.2)
    #plt.colorbar(im, cax=cax)

    fs_in_fig = 5
    colorbar_pad = 0.1
    colorbar_size = '2%'

    # upper area
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize_fc)
    cmap = cm.Reds
    norm = cm.colors.Normalize(vmax=upper_area_map.max(), vmin=upper_area_map.min())
    im = ax.imshow(upper_area_map, aspect='equal', origin='lower', cmap=cmap, norm=norm)
    ax.set_yticks(range(len(b_list)))
    ax.set_xticks(range(len(a_list))[::2])
    ax.set_yticklabels([ '%.2f'%val for val in b_list ])
    ax.set_xticklabels([ '%.2f'%val for val in a_list[::2] ])
    ax.set_xlabel(r'$\beta_\mathrm{I}$')
    ax.set_ylabel(r'$\beta_\mathrm{E}$')
    for i in range(len(b_list)):
        for j in range(len(a_list)):
            text = ax.text(j, i, "%.3f"%upper_area_map[i, j],
                           ha="center", va="center", color="k", size=fs_in_fig)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=colorbar_size, pad=colorbar_pad)
    plt.colorbar(im, cax=cax, label='Upper Area')

    # lower area
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize_fc)
    #cmap = cm.Blues
    cmap = cm.Blues_r
    norm = cm.colors.Normalize(vmax=lower_area_map.max(), vmin=lower_area_map.min())
    im = ax.imshow(lower_area_map, aspect='equal', origin='lower', cmap=cmap, norm=norm)
    ax.set_yticks(range(len(b_list)))
    ax.set_xticks(range(len(a_list))[::2])
    ax.set_yticklabels([ '%.2f'%val for val in b_list ])
    ax.set_xticklabels([ '%.2f'%val for val in a_list[::2] ])
    ax.set_xlabel(r'$\beta_\mathrm{I}$')
    ax.set_ylabel(r'$\beta_\mathrm{E}$')
    for i in range(len(b_list)):
        for j in range(len(a_list)):
            text = ax.text(j, i, "%.3f"%lower_area_map[i, j],
                           ha="center", va="center", color="k", size=fs_in_fig)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=colorbar_size, pad=colorbar_pad)
    plt.colorbar(im, cax=cax, label='Lower Area')

    # summary
    area_max = upper_area_map
    indices = np.where(upper_area_map<(-lower_area_map))
    area_max[indices] = lower_area_map[indices]
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize_fc)
    cmap = cm.RdBu_r
    norm = cm.colors.Normalize(vmax=np.abs(area_max).max(), vmin=-np.abs(area_max).max())
    im = ax.imshow(area_max, aspect='equal', origin='lower', cmap=cmap, norm=norm)
    ax.set_yticks(range(len(b_list)))
    ax.set_xticks(range(len(a_list))[::2])
    ax.set_yticklabels([ '%.2f'%val for val in b_list ])
    ax.set_xticklabels([ '%.2f'%val for val in a_list[::2] ])
    ax.set_xlabel(r'$\beta_\mathrm{I}$')
    ax.set_ylabel(r'$\beta_\mathrm{E}$')
    for i in range(len(b_list)):
        for j in range(len(a_list)):
            text = ax.text(j, i, "%.3f"%area_max[i, j],
                           ha="center", va="center", color="k", size=fs_in_fig)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=colorbar_size, pad=colorbar_pad)
    plt.colorbar(im, cax=cax, label=r'$\Delta q$')

    fig, ax = plt.subplots(figsize=figsize_sc)
    for i, b in enumerate(b_list):
        ax.plot(a_list, area_max[i], label=r'$\beta_\mathrm{E}=%.2f$'%b)
    ax.axhline(0, color='k', ls='--')
    ax.set_xticks(a_list[::2])
    ax.set_xticklabels([ '%.2f'%val for val in a_list[::2] ])
    ax.set_xlabel(r'$\beta_\mathrm{I}$')
    ax.set_ylabel(r'$\Delta q$')
    plt.legend(fontsize=legend_size)
    plt.tight_layout()

if outfile:
    pdf = PdfPages(outfile)
    fignums = plt.get_fignums()
    for fignum in fignums:
        plt.figure(fignum)
        pdf.savefig()
    pdf.close()
if plot:
    plt.show()



import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from math import pi, cos, sin

plt.rcParams.update({
    "font.size": 8, 
    "xtick.labelsize": 7,
    "ytick.labelsize": 7
    }) 
legend_size = 7
figsize_fc = (7.48, 2.0)
figsize_sc = (3.54, 2.0)
figsize_hc = (2.0, 1.5)

parser = argparse.ArgumentParser()
parser.add_argument('weight',
    type=str, help='weight file (npz)')
parser.add_argument('-o', '--outfile',
    type=str, default=None, help='file name (pdf)')
parser.add_argument('-p', '--plot',
    action='store_true', help='plot if stored')
parser.add_argument('-v', '--visualize',
    action='store_true', help='visualize connections')
args = parser.parse_args()

w_file = args.weight
outfile = args.outfile
plot = args.plot
visualize = args.visualize

params = np.load(w_file, 'r')
N_exc, N_inh = params['w_EI'].shape
N = N_exc+N_inh

fig, axs = plt.subplots(ncols=4, figsize=figsize_fc)
for i, direction in enumerate(('EE', 'EI', 'IE', 'II')):
    W = params['w_'+direction]
    sources, targets = np.where(np.isnan(W)==False)

    bins = np.linspace(0, 1, int(np.sqrt(sources.shape[0])))
    hist, bin_edges = np.histogram(W[sources, targets], bins=bins)

    axs[i].bar(bin_edges[:-1], hist, width=bin_edges[1]-bin_edges[0], color='k')
    axs[i].set_xlabel(r'Weight $w$')
axs[0].set_ylabel('Counts')
plt.tight_layout()

fig, axs = plt.subplots(nrows=3, figsize=(7.48, 5), sharex=True)
weight_matrix = np.zeros((N, N), dtype=np.float32)
weight_matrix[0:N_exc, 0:N_exc] = params['w_EE']
weight_matrix[0:N_exc, N_exc:N] = params['w_EI']
weight_matrix[N_exc:N, 0:N_exc] = params['w_IE']
weight_matrix[N_exc:N, N_exc:N] = params['w_II']
sources, targets = np.where(np.isnan(weight_matrix)==True)
weight_matrix[sources, targets] = 0
fan_in_exc = np.sum(weight_matrix[:N_exc], axis=0)
fan_in_inh = np.sum(weight_matrix[N_exc:], axis=0)
fan_in = fan_in_exc+fan_in_inh
axs[0].bar(range(1, N+1), fan_in_exc, color='r')
axs[0].bar(range(1, N+1), fan_in_inh, color='b', bottom=fan_in_exc)
axs[0].set_xlim(0, N+1)
axs[0].set_ylabel('Total Input Weight')

weight_matrix = np.zeros((N, N), dtype=np.float32)
weight_matrix[0:N_exc, 0:N_exc] = params['w_EE']
weight_matrix[0:N_exc, N_exc:N] = params['w_EI']
weight_matrix[N_exc:N, 0:N_exc] = params['w_IE']
weight_matrix[N_exc:N, N_exc:N] = params['w_II']
sources, targets = np.where(np.isnan(weight_matrix)==True)
weight_matrix[sources, targets] = 0
fan_out_exc = np.sum(weight_matrix[:, :N_exc], axis=1)
fan_out_inh = np.sum(weight_matrix[:, N_exc:], axis=1)
fan_out = fan_out_exc+fan_out_inh
axs[1].bar(range(1, N+1), fan_out_exc, color='r')
axs[1].bar(range(1, N+1), fan_out_inh, color='b', bottom=fan_out_exc)
axs[1].set_xlim(0, N+1)
axs[1].set_ylabel('Total Output Weight')

weight_matrix = np.zeros((N, N), dtype=np.float32)
weight_matrix[0:N_exc, 0:N_exc] = params['w_EE']
weight_matrix[0:N_exc, N_exc:N] = params['w_EI']
weight_matrix[N_exc:N, 0:N_exc] = params['w_IE']
weight_matrix[N_exc:N, N_exc:N] = params['w_II']
sources, targets = np.where(np.isnan(weight_matrix)==True)
weight_matrix[sources, targets] = 0
weight_sum = np.sum(weight_matrix, axis=0)
ratio_exc = np.sum(weight_matrix[0:N_exc], axis=0)/weight_sum*100
ratio_inh = np.sum(weight_matrix[N_exc:N], axis=0)/weight_sum*100
axs[2].bar(range(1, N+1), ratio_exc, color='r')
axs[2].bar(range(1, N+1), ratio_inh, color='b', bottom=ratio_exc)
axs[2].set_xlim(0, N+1)
axs[2].set_xlabel('Neuron')
axs[2].set_ylabel('EI Ratio of Input Weight, %')
fig.align_ylabels(axs)
plt.tight_layout()

fig, ax = plt.subplots(figsize=figsize_sc)
ax.plot(fan_in[:N_exc], fan_out[:N_exc], '.r', alpha=0.2)
ax.plot(fan_in[N_exc:], fan_out[N_exc:], '.b', alpha=0.2)
ax.set_xlabel('Total Input Weight')
ax.set_ylabel('Total Output Weight')
plt.tight_layout()

fig, ax = plt.subplots(figsize=figsize_sc)
ax.plot(fan_in[:N_exc], ratio_exc[:N_exc], '.r', alpha=0.2)
ax.plot(fan_in[N_exc:], ratio_exc[N_exc:], '.b', alpha=0.2)
ax.set_xlabel('Total Input Weight')
ax.set_ylabel('EI Ratio of Input Weight, %')
plt.tight_layout()

#fig, axs = plt.subplots(nrows=4, figsize=(12, 7), sharex=True)
#for i, direction in enumerate(('EE', 'EI', 'IE', 'II')):
#    W = params['w_'+direction]
#    sources, targets = np.where(np.isnan(W)==True)
#    W[sources, targets] = 0
#    print(W.shape)
#    axs[i].bar(range(1, W.shape[1]+1), np.sum(W, axis=0), color='k')
#axs[-1].set_xlim(0, max(N-N_exc, N_exc)+1)
#plt.tight_layout()
#
#fig, axs = plt.subplots(nrows=4, figsize=(12, 7), sharex=True)
#for i, direction in enumerate(('EE', 'EI', 'IE', 'II')):
#    W = params['w_'+direction]
#    sources, targets = np.where(np.isnan(W)==True)
#    W[sources, targets] = 0
#    print(W.shape)
#    axs[i].bar(range(1, W.shape[0]+1), np.sum(W, axis=1), color='k')
#axs[-1].set_xlim(0, max(N-N_exc, N_exc)+1)
#plt.tight_layout()

#fig, ax = plt.subplots(ncols=1, figsize=(12, 3))
#weight_matrix = np.zeros((N, N), dtype=np.float32)
#weight_matrix[0:N_exc, 0:N_exc] = params['w_EE']
#weight_matrix[0:N_exc, N_exc:N] = params['w_EI']
#weight_matrix[N_exc:N, 0:N_exc] = -params['w_IE']
#weight_matrix[N_exc:N, N_exc:N] = -params['w_II']
#sources, targets = np.where(np.isnan(weight_matrix)==True)
#weight_matrix[sources, targets] = 0
#ax.bar(range(1, N+1), np.sum(weight_matrix, axis=0), color='k')
#ax.set_xlim(0, N+1)
#plt.tight_layout()

if visualize:
    fig, ax = plt.subplots(figsize=(1.3, 1.3))
    for direction in ('EE', 'EI', 'IE', 'II'):
        weight_matrix = params['w_'+direction]
        sources, targets = np.where(np.isnan(weight_matrix)==False)
        strengths = [ weight_matrix[source, target] for source, target, in zip(sources, targets) ]
        strengths = np.array(strengths)
        indices = np.where(strengths>=0.5)[0]
        sources = sources[indices]
        targets = targets[indices]
        strengths = strengths[indices]

        cm_name = 'Reds'
        if direction[0] == 'I':
            sources += N_exc
            cm_name = 'Blues'
        if direction[1] == 'I':
            targets += N_exc
        cm = plt.get_cmap(cm_name)
        for source, target, strength in zip(sources, targets, strengths):
            rad_s = 2*pi*source/N
            rad_t = 2*pi*target/N
            ax.plot(
                [cos(rad_s), cos(rad_t)],
                [sin(rad_s), sin(rad_t)], c=cm(strength), alpha=0.3, lw=0.2)
    for i in range(N_exc):
        rad = 2*pi*i/N
        ax.plot(cos(rad), sin(rad), 'or', ms=0.6)
    for i in range(N_exc, N):
        rad = 2*pi*i/N
        ax.plot(cos(rad), sin(rad), 'ob', ms=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')

    fig, axs = plt.subplots(ncols=2, figsize=(0.8, 1.3))
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    #for ax, cm_name in zip(axs, ['Reds', 'Blues']):
    #    mpl.colorbar.ColorbarBase(ax, cmap=plt.get_cmap(cm_name),
    #        orientation='vertical', ticks=np.linspace(0, 1, 5))
    cb_r = mpl.colorbar.ColorbarBase(axs[0], cmap=plt.get_cmap('Reds'),
        orientation='vertical', ticks=np.linspace(0, 1, 5))
    cb_r.set_ticklabels([])
    cb_b = mpl.colorbar.ColorbarBase(axs[1], cmap=plt.get_cmap('Blues'),
        orientation='vertical', ticks=np.linspace(0, 1, 5), label=r'Weight $w$')
    cb_b.set_ticklabels([ '%.1f'%val for val in np.linspace(0, 1, 5) ])
    for ax in axs: ax.tick_params(labelsize=6, length=1.5)
    #cbar_r = mpl.colorbar.ColorbarBase(axs[0], cmap='Reds',
    #    norm=norm, orientation='vertical', ticks=np.linspace(0, 1, 5))
    #cbar_b = mpl.colorbar.ColorbarBase(axs[1], cmap='Blues',
    #    norm=norm, orientation='vertical', ticks=np.linspace(0, 1, 5))
    plt.tight_layout(w_pad=0.01)

if outfile:
    pdf = PdfPages(outfile)
    fignums = plt.get_fignums()
    for fignum in fignums:
        plt.figure(fignum)
        pdf.savefig()
    pdf.close()
if plot: plt.show()


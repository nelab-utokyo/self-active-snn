
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

parser = argparse.ArgumentParser()
parser.add_argument('config',
    type=str, help='config file (json)')
parser.add_argument('record',
    type=str, help='record file (npz)')
parser.add_argument('-o', '--outfile',
    type=str, default=None, help='file name (pdf)')
parser.add_argument('-p', '--plot',
    action='store_true', help='plot if stored')
args = parser.parse_args()

c_file = args.config
r_file = args.record
outfile = args.outfile
plot = args.plot

N, N_exc = None, None
with open(c_file, 'r') as f:
    data = json.load(f)
    N = data['N']
    N_exc = int(data['N']*(1.-data['inhibitory_ratio']))

length = None
spike_i, spike_t = None, None
state_t, state_v = None, None
gtot_exc, gtot_inh = None, None
trace_g = {}
trace_x = {}
trace_w = {}
with np.load(r_file, 'r') as data:
    length = data['length']
    spike_i, spike_t = data['spiketimes']
    state_t, state_v = data['time'], data['mua']
    gtot_exc, gtot_inh = data['gtot_exc'], data['gtot_inh']
    for direction in ('EE', 'EI', 'IE', 'II'):
        trace_g[direction] = data['trace_g_'+direction]
        trace_x[direction] = data['trace_x_'+direction]
        trace_w[direction] = data['trace_w_'+direction]
min_gtot = min(np.min(gtot_exc), np.min(gtot_inh))
max_gtot = max(np.max(gtot_exc), np.max(gtot_inh))

# Plot states
extent = [state_t[0], length, 0, N]

fig, axs = plt.subplots(nrows=3, figsize=(12, 7), sharex=True)
axs[0].plot(spike_t, spike_i, '.k', ms=1)
im = axs[1].imshow(state_v, aspect='auto', origin='lower', extent=extent)
axs[2].plot(state_t, np.mean(state_v[:N_exc], axis=0), c='r', lw=1)
axs[2].plot(state_t, np.mean(state_v[N_exc:], axis=0), c='b', lw=1)
axs[2].set_xlim(0, length)
fig.colorbar(im, ax=axs)

fig, axs = plt.subplots(nrows=3, figsize=(12, 7), sharex=True)
im0 = axs[0].imshow(gtot_exc, vmin=min_gtot, vmax=max_gtot,
    aspect='auto', origin='lower', extent=extent)
im1 = axs[1].imshow(gtot_inh, vmin=min_gtot, vmax=max_gtot,
    aspect='auto', origin='lower', extent=extent)
axs[2].plot(state_t, np.mean(gtot_exc, axis=0), c='r', lw=1)
axs[2].plot(state_t, np.mean(gtot_inh, axis=0), c='b', lw=1)
axs[2].set_xlim(0, length)
fig.colorbar(im0, ax=axs)

for direction in ('EE', 'EI', 'IE', 'II'):
    fig, axs = plt.subplots(nrows=2, figsize=(12, 3), sharex=True)
    im = axs[0].imshow(trace_g[direction], aspect='auto', origin='lower')
    axs[1].plot(np.mean(trace_g[direction], axis=0), c='k', lw=1)
    axs[1].set_xlim(0, len(state_t))
    fig.colorbar(im, ax=axs)
    plt.suptitle(direction)

for direction in ('EE', 'EI', 'IE', 'II'):
    fig, axs = plt.subplots(nrows=2, figsize=(12, 3), sharex=True)
    im = axs[0].imshow(trace_x[direction], aspect='auto', origin='lower')
    axs[1].plot(np.mean(trace_x[direction], axis=0), c='k', lw=1)
    axs[1].set_xlim(0, len(state_t))
    fig.colorbar(im, ax=axs)
    plt.suptitle(direction)

for direction in ('EE', 'EI', 'IE', 'II'):
    fig, axs = plt.subplots(nrows=2, figsize=(12, 3), sharex=True)
    im = axs[0].imshow(trace_w[direction], aspect='auto', origin='lower')
    axs[1].plot(np.mean(trace_w[direction], axis=0), c='k', lw=1)
    axs[1].set_xlim(0, len(state_t))
    fig.colorbar(im, ax=axs)
    plt.suptitle(direction)

if outfile:
    pdf = PdfPages(outfile)
    fignums = plt.get_fignums()
    for fignum in fignums:
        plt.figure(fignum)
        pdf.savefig()
    pdf.close()
if plot:
    plt.show()



import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from brian2 import second
from lib.utils import calc_cv
from lib.avalanche import get_avalanche_sizes, get_counts, fit_powerlaw, fit_expon
from lib.utils import linear_regression

plt.rcParams.update({
    "font.size": 8, 
    "xtick.labelsize": 7,
    "ytick.labelsize": 7
    }) 
legend_size = 7
figsize_fc = (7.48, 2)
figsize_lg = (3.74, 1.5)
figsize_hc = (2.0, 1.5)

parser = argparse.ArgumentParser()
parser.add_argument('config',
    type=str, help='config file (json)')
parser.add_argument('record',
    type=str, help='record file (npz)')
parser.add_argument('-o', '--outfile',
    type=str, default=None, help='file name (pdf)')
parser.add_argument('-l', '--length',
    type=int, default=120, help='length (second)')
parser.add_argument('-p', '--plot',
    action='store_true', help='plot if stored')
args = parser.parse_args()

c_file = args.config
r_file = args.record
outfile = args.outfile
length = args.length
plot = args.plot

N, N_exc = None, None
with open(c_file, 'r') as f:
    data = json.load(f)
    N = data['N']
    N_exc = int(data['N']*(1.-data['inhibitory_ratio']))

spike_i, spike_t = None, None
with np.load(r_file, 'r') as data:
    spike_i, spike_t = data['spiketimes']

# Raster Plot
indices = np.where(spike_t<length)[0]
fig, ax = plt.subplots(nrows=1, figsize=figsize_lg)
ax.plot(spike_t[indices], spike_i[indices], 'ok', ms=0.2)
ax.set_ylabel('Neuron')
ax.set_xlabel('Time, second')
ax.set_xlim(0, length)
ax.set_ylim(0, N)
ax.set_yticks(np.linspace(0, N, 6))
plt.tight_layout()

# Plot Activity statistics
avalanche_sizes = get_avalanche_sizes(spike_t, unit=second, sort=True)
sizes, counts = get_counts(avalanche_sizes)

fig, axs = plt.subplots(ncols=3, figsize=figsize_lg)
axs[0].plot(sizes, counts, '.k', ms=1)
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[1].plot(sizes, counts, '.k', ms=1)
axs[1].set_yscale('log')
axs[2].plot(sizes, counts, '.k', ms=1)
axs[0].set_ylabel('Counts')
for ax in axs: ax.set_xlabel('Avalanche Size')
plt.tight_layout()

fig, ax = plt.subplots(nrows=1, figsize=figsize_hc)
ax.plot(sizes, counts.astype(np.float32)/np.sum(counts), '.k', ms=1)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('Probability')
ax.set_xlabel('Avalanche Size')
plt.tight_layout()

# probability density fitting
x = np.arange(1, N+1)
# fit exponential
lam, c_expon, LL_lambda = fit_expon(avalanche_sizes, 1, N)
y_expon = c_expon*np.exp(-lam*x)
# fit powerlaw
alpha, c_power, LL_alpha = fit_powerlaw(avalanche_sizes, 1, N)
y_power = c_power*np.power(x, alpha)
# plot estimates
fig, ax = plt.subplots(nrows=1, figsize=figsize_hc)
rates = counts.astype(np.float32)/np.sum(counts)
ax.plot(sizes, rates, '.k', ms=1)
ax.plot(x, y_expon, c='b', ls='-', lw=0.8)
ax.plot(x, y_power, c='r', ls='-', lw=0.8)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.9, np.max(avalanche_sizes)*1.2)
ax.set_ylim(np.min(rates)*0.5, np.max(rates)*2)
ax.set_ylabel('Probability')
ax.set_xlabel('Avalanche Size')
plt.tight_layout()

fig, ax = plt.subplots(nrows=1, figsize=figsize_hc)
rates = counts.astype(np.float32)/np.sum(counts)
ax.plot(sizes, rates, '.k', ms=1)
ax.plot(x, y_expon, c='b', ls='-', lw=0.8)
ax.plot(x, y_power, c='r', ls='-', lw=0.8)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.9, np.max(avalanche_sizes)*1.2)
ax.set_ylim(np.min(rates)*0.5, np.max(rates)*2)
ax.set_ylabel('Probability')
ax.set_xlabel('Avalanche Size')
#ax.set_title(r'$\alpha=%.3f$, $\lambda=%.3f : %.1f$'%(alpha, lam, LL_alpha-LL_lambda))
ax.set_title(r'$\alpha=%.3f$, $\lambda=%.3f$'%(alpha, lam))
plt.tight_layout()

# delta q
indices = np.where((avalanche_sizes>=1)&(avalanche_sizes<=N))[0]
avalanche_sizes = avalanche_sizes[indices]
sizes, counts = get_counts(avalanche_sizes)

sizes_log10, counts_log10 = np.log10(sizes).reshape((-1, 1)), np.log10(counts).reshape((-1, 1))
x_lreg = np.log10(range(1, N+1)).reshape((-1, 1))
y_emp, y_lreg, range_best = linear_regression(sizes_log10, counts_log10, x_lreg)

p_emp = np.zeros((N, 1))
total = np.sum(counts).astype(np.float32)
for size, count in zip(sizes, counts):
    p_emp[size-1, 0] = float(count)/total
p_the = np.power(10, y_lreg)
p_the = p_the/total

#diff = p_emp[1:, 0]-p_the[1:, 0]
diff = p_emp[range_best[1]:, 0]-p_the[range_best[1]:, 0]
indices = np.where(diff>0)[0]
upper_area = np.sum(diff[indices])
indices = np.where(diff<0)[0]
lower_area = np.sum(diff[indices])
delta = upper_area if upper_area >= (-lower_area) else lower_area

x = np.arange(1, N+1)
fig, ax = plt.subplots(ncols=1, figsize=figsize_hc)
rates = counts.astype(np.float32)/np.sum(counts)
#y1, y2 = p_emp[1:, 0], p_the[1:, 0]
y1, y2 = p_emp[range_best[1]:, 0], p_the[range_best[1]:, 0]
ax.plot(sizes, rates, '.k', ms=1)
ax.plot(x, p_emp, c='k', ls='-', lw=0.6)
ax.plot(x, p_the, c='grey', ls='-', lw=0.8)
ax.fill_between(x[range_best[1]:], y1, y2, where=(y1>y2), color='r', alpha=0.4)
ax.fill_between(x[range_best[1]:], y1, y2, where=(y1<y2), color='b', alpha=0.4)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.9, np.max(avalanche_sizes)*1.2)
ax.set_ylim(np.min(rates)*0.5, np.max(rates)*2)
ax.set_ylabel('Probability')
ax.set_xlabel('Avalanche Size')
plt.tight_layout()

fig, ax = plt.subplots(ncols=1, figsize=figsize_hc)
rates = counts.astype(np.float32)/np.sum(counts)
#y1, y2 = p_emp[1:, 0], p_the[1:, 0]
y1, y2 = p_emp[range_best[1]:, 0], p_the[range_best[1]:, 0]
ax.plot(sizes, rates, '.k', ms=1)
ax.plot(x, p_emp, c='k', ls='-', lw=0.6)
ax.plot(x, p_the, c='grey', ls='-', lw=0.8)
ax.fill_between(x[range_best[1]:], y1, y2, where=(y1>y2), color='r', alpha=0.4)
ax.fill_between(x[range_best[1]:], y1, y2, where=(y1<y2), color='b', alpha=0.4)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.9, np.max(avalanche_sizes)*1.2)
ax.set_ylim(np.min(rates)*0.5, np.max(rates)*2)
ax.set_ylabel('Probability')
ax.set_xlabel('Avalanche Size')
fig.suptitle(r'$\Delta q=%.3f$'%delta)
plt.tight_layout()


cvs = calc_cv(spike_t, spike_i, N)
hist, bin_edges = np.histogram(cvs, bins='sqrt')
fig, ax = plt.subplots(nrows=1, figsize=figsize_hc)
ax.bar(bin_edges[:-1], hist, width=bin_edges[1]-bin_edges[0], color='k')
ax.set_ylabel('Counts')
ax.set_xlabel('CV')
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



import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from brian2 import *

plt.rcParams.update({
    "font.size": 8, 
    "xtick.labelsize": 7,
    "ytick.labelsize": 7
    }) 
legend_size = 6
figsize = (4.98, 2.2)
w_pad = 2

parser = argparse.ArgumentParser()
parser.add_argument('--betas',
    type=float, nargs='+', default=[1.0, 1.2])
parser.add_argument('--tau_estdp',
    type=float, default=20)
parser.add_argument('--delta_estdp',
    type=float, default=0.02)
parser.add_argument('--tau_1',
    type=float, default=10.)
parser.add_argument('--tau_2',
    type=float, default=20.)
parser.add_argument('--delta_istdp',
    type=float, default=0.02)
parser.add_argument('-o', '--outfile',
    type=str, default=None, help='file name (pdf)')
parser.add_argument('-p', '--plot',
    action='store_true', help='plot if stored')
args = parser.parse_args()

# excitatory STDP
betas = args.betas
tau_estdp = args.tau_estdp*ms
delta_estdp = args.delta_estdp
# inhibitory STDP
tau_1 = args.tau_1*ms
tau_2 = args.tau_2*ms
delta_istdp = args.delta_istdp
# save and plot
outfile = args.outfile
plot = args.plot

N = 101
tmax = 100*ms

eqs_synapse_E = '''
dapre/dt = -apre/tau_estdp : 1 (event-driven)
dapost/dt = -apost/tau_estdp : 1 (event-driven)
w : 1
'''

eqs_synapse_I = '''
da1pre/dt = -a1pre/tau_1 : 1 (event-driven)
da2pre/dt = -a2pre/tau_2 : 1 (event-driven)
da1post/dt = -a1post/tau_1 : 1 (event-driven)
da2post/dt = -a2post/tau_2 : 1 (event-driven)
w : 1
'''

on_pre_E = '''
apre += delta_pre
w = w+apost
'''

on_post_E = '''
apost -= delta_post
w = w+apre
'''

on_pre_I = '''
a1pre += delta_1
a2pre += delta_2
w = w+a1post-a2post
'''

on_post_I = '''
a1post += delta_1
a2post += delta_2
w = w+a1pre-a2pre
'''

G = NeuronGroup(N, 'tspike:second', threshold='t>tspike', refractory=tmax*2)
H = NeuronGroup(N, 'tspike:second', threshold='t>tspike', refractory=tmax*2)
G.tspike = 'i*tmax/(N-1)'
H.tspike = '(N-1-i)*tmax/(N-1)'

S_E = Synapses(G, H,
             eqs_synapse_E,
             on_pre=on_pre_E,
             on_post=on_post_E)
S_E.connect(j='i')

S_I = Synapses(G, H,
             eqs_synapse_I,
             on_pre=on_pre_I,
             on_post=on_post_I)
S_I.connect(j='i')

store()

fig, axs = plt.subplots(ncols=2, figsize=figsize)
axs[0].axhline(0, ls='-', lw=0.5, c='k')
axs[0].axvline(0, ls='-', lw=0.5, c='k')
axs[1].axhline(0, ls='-', lw=0.5, c='k')
axs[1].axvline(0, ls='-', lw=0.5, c='k')
for beta in betas:
    restore()
    delta_pre, delta_post = delta_estdp, beta*delta_estdp
    delta_1 = delta_istdp/(1.-(beta*tau_1/tau_2))
    delta_2 = delta_1*beta*tau_1/tau_2

    run(tmax+1*ms)

    print("%.2f"%beta, np.sum(S_E.w))
    print("%.2f"%beta, np.sum(S_I.w))

    axs[0].plot((H.tspike-G.tspike)/ms, S_E.w, lw=0.8, label=r"$\beta_\mathrm{E}=%.1f$"%beta)
    axs[1].plot((H.tspike-G.tspike)/ms, S_I.w, lw=0.8, label=r"$\beta_\mathrm{I}=%.1f$"%beta)
axs[0].set_xlabel(r'$\Delta t$, ms')
axs[0].set_ylabel(r'$\Delta w_\mathrm{E\cdot}$')
axs[0].legend(fontsize=legend_size)
axs[1].set_xlabel(r'$\Delta t$, ms')
axs[1].set_ylabel(r'$\Delta w_\mathrm{I\cdot}$')
axs[1].legend(fontsize=legend_size)
plt.tight_layout(w_pad=w_pad)

if outfile:
    pdf = PdfPages(outfile)
    fignums = plt.get_fignums()
    for fignum in fignums:
        plt.figure(fignum)
        pdf.savefig()
    pdf.close()
if plot: plt.show()


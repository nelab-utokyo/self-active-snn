
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import sem, f, friedmanchisquare, wilcoxon, binom_test

plt.rcParams.update({
    "font.size": 8, 
    "xtick.labelsize": 7,
    "ytick.labelsize": 7
    }) 
legend_size = 7
figsize_fc = (7.48, 2)
figsize_sc = (3.54, 2.0)

#def welch_anova(X):
#    k = len(X)
#    counts = np.array([ len(x) for x in X ]).astype(np.float64)                         
#    means = np.array([ np.mean(x) for x in X ]).astype(np.float64)                      
#    variances = np.array([ np.var(x) for x in X ]).astype(np.float64)                   
#    w_j = counts/variances
#    w = np.mean(counts)
#    x_dash_bar = np.dot(w_j, means)/w                                                   
#
#    a_bar = np.dot(w_j, np.square(means-x_dash_bar))                                    
#    b_bar = np.sum(np.square(1-w_j/w)/(counts-1))                                       
#
#    F = a_bar/((k-1)*(1+2*(k-2)/(k*k-1)*b_bar))                                         
#    df1 = k-1
#    df2 = (k*k-1)/(3*b_bar)
#    p_value = f.sf(F, df1, df2)
#
#    return F, df1, df2, p_value

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
        dataset[condition] = {'accs': [], 'coefs': []}

    #print(infile)
    with np.load(infile, 'r') as data:
        i = np.where(data['Cs']==10)[0]
        dataset[condition]['n_splits'] = data['n_splits']

        #bin_edges = data['bin_edges']
        bin_edges = np.around(data['bin_edges'], 4)
        indices_bin = np.where(bin_edges<=length*1.1)[0]
        dataset[condition]['bin_edges'] = bin_edges[indices_bin]

        dataset[condition]['accs'].append(data['accs'][indices_bin, i].tolist())
        dataset[condition]['coefs'].append(data['coefs'][indices_bin, i].tolist())
        if len(infile.split('/')[-1].split('_')) == 4: print(condition, data['accs'][indices_bin, i].shape, infile)
if conditions is None: conditions = list(dataset.keys())

step = 15
bin_edges = dataset[conditions[0]]['bin_edges']
n_splits = dataset[conditions[0]]['n_splits']
n_classes = np.array(dataset[conditions[0]]['coefs']).shape[3]

gradient = np.linspace(0.0, 0.8, len(conditions))
cmap = plt.get_cmap('viridis')
#cmap = plt.get_cmap('jet')

fig, axs = plt.subplots(nrows=2, figsize=figsize_sc, sharex=True)
data = []
for i, condition in enumerate(conditions):
    accs = np.array(dataset[condition]['accs'])
    #mean_accs = np.mean(accs, axis=(0, 2))
    mean_accs = np.mean(accs, axis=2)
    trace = np.median(mean_accs, axis=0)
    color = cmap(gradient[i])

    axs[0].axhline(1./n_classes, c='k', ls='--', lw=0.2)
    axs[0].plot(bin_edges, trace,
        c=color, alpha=1., label=dict_ABB[condition], lw=1)

    data.append(np.mean(accs, axis=2).tolist())
data = np.array(data).T
p_values = []
for data_each in data:
    #p_value = welch_anova(data_each.T)[3]
    p_value = friedmanchisquare(*data_each.T)[1]
    p_values.append(p_value)
#p_values = np.array(p_values)*len(data)
p_values = np.array(p_values)
axs[0].set_ylabel('Accuracy')
#axs[0].set_yscale('log')
axs[0].legend(fontsize=legend_size, ncol=1)
indices = np.where(bin_edges>0)[0]
axs[1].plot(bin_edges[indices], p_values[indices], c='k', lw=1)
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
for i, condition in enumerate(conditions):
    coefs = np.array(dataset[condition]['coefs'])
    color = cmap(gradient[i])

    ax.plot(bin_edges, np.mean(coefs, axis=(0, 2, 3)),
        c=color, alpha=1., label=dict_ABB[condition], lw=1)
ax.set_xticks(np.arange(-0.04, length+0.04, 0.04))
ax.xaxis.set_minor_locator(MultipleLocator(0.02))
ax.set_xlim(-0.04, length)
ax.set_xlabel('Time, second')
ax.set_ylabel('#Coefficients')
plt.legend(fontsize=legend_size, ncol=1)
plt.tight_layout()

#indices_bin = np.where((bin_edges>0.05)&(bin_edges<=length*1.1))[0]
#indices_bin = np.where((bin_edges>0.05)&(p_values<=0.001))[0]
#indices_bin = np.where(p_values<p_threshold)[0]
indices_bin = np.where((bin_edges>0.0)&(p_values<=p_threshold))[0]
if len(indices_bin) > 0:
    indices_bin = indices_bin[np.argmin(p_values[indices_bin])]
    print(indices_bin)
    print(bin_edges[indices_bin])
    results = []
    for i, condition in enumerate(conditions):
        accs = np.array(dataset[condition]['accs'])
        accs = accs[:, indices_bin]
        #mean_accs = np.mean(accs, axis=(1, 2)).tolist()
        mean_accs = np.mean(accs, axis=1).tolist()
        results.append(mean_accs)
    results = np.array(results)

    #stat, p_value = friedmanchisquare(*results.tolist())
    #print(p_value)

    fig, ax = plt.subplots(figsize=figsize_sc)
    for i, data_each in enumerate(results, 1):
        jitter = 0.1*np.random.random(len(data_each))-0.05
        ax.plot(jitter+i, data_each, '.', c='k', alpha=0.2)
    ax.boxplot(results.tolist(), labels=labels, whis=[0, 100], widths=0.8)
    ax.set_ylabel('Mean Accuracy')
    plt.tight_layout()

    fig, ax = plt.subplots(figsize=figsize_sc)
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

    fig, ax = plt.subplots(figsize=figsize_sc)
    vmin, vmax = 1, 0
    for i, data_each in enumerate(results, 1):
        jitter = 0.1*np.random.random(len(data_each))-0.05
        ax.plot(jitter+i, data_each, '.', c='k', alpha=0.2)
        vmin = min(vmin, np.min(data_each))
        vmax = max(vmax, np.max(data_each))
    ax.boxplot(results.tolist(), labels=labels, whis=[0, 100], widths=0.8)
    ax.set_ylim(top=vmax+(vmax-vmin)*0.3*np.max(n_significant))
    ax.set_ylabel('Mean Accuracy')
    plt.tight_layout()

decay_times = []
ind_bin = np.where(bin_edges > 0)[0]
for t, condition in enumerate(conditions):
    accs = np.array(dataset[condition]['accs'])

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
ax.boxplot(decay_times, labels=labels, whis=[0, 100])
ax.grid(axis='y')
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
#ax.set_xlabel('Conditions')
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


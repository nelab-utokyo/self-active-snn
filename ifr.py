
import argparse
import numpy as np

def burst_detection(spiketimes, length, th=5.):
    counts = 0
    for st in spiketimes:
        counts += len(st)
    return counts > int(len(spiketimes)*length*th)

def to_ifr(spiketimes, wwidth, wshift, t_stop, t_start=0):
    b_starts = np.arange(t_start, t_stop-wwidth+wshift, wshift)
    ifr = []
    for st in spiketimes:
        st_np = np.array(st)
        bst = []
        for b_start in b_starts:
            count = (st_np >= b_start)*(st_np < (b_start+wwidth))
            count = np.sum(count)/wwidth
            bst.append(count)
        ifr.append(bst)
    return b_starts+wwidth, ifr

parser = argparse.ArgumentParser()
parser.add_argument('records',
    type=str, nargs='+',
    help='record files (npz)')
parser.add_argument('-o', '--outfile',
    type=str, default=None,
    help='outfilename (npz)')
parser.add_argument('-w', '--wwidth',
    type=float, default=0.01,
    help='window width (sec)')
parser.add_argument('-s', '--wshift',
    type=float, default=0.01,
    help='window shift (sec)')
parser.add_argument('--offset',
    type=float, default=0.001,
    help='offset from stimulation time (sec)')
parser.add_argument('-p', '--plot',
    action='store_true', help='plot')
parser.add_argument('--pre',
    type=float, default=0.1, help='pre (second)')
parser.add_argument('--post',
    type=float, default=None, help='post (second)')
args = parser.parse_args()

records = args.records
outfile = args.outfile
wwidth = args.wwidth
wshift = args.wshift
offset = args.offset
plot = args.plot
pre = args.pre
post = args.post

X = []
y = []
bin_edges = None
for record in records:
    spiketimes = []
    length = None
    with np.load(record, 'r') as data:
        spike_i, spike_t = data['spiketrain']
        _, input_t = data['input']
        N = data['N']
        whole_length = data['length']

        stim_end = np.max(input_t)
        spike_t = spike_t-stim_end

        indices = np.where(spike_t>=(-pre))[0]
        spike_i = spike_i[indices].astype(np.int16)
        spike_t = spike_t[indices]

        if post is not None:
            indices = np.where(spike_t<=post)[0]
            spike_i = spike_i[indices].astype(np.int16)
            spike_t = spike_t[indices]

        for i in range(N):
            indices = np.where(spike_i == i)[0]
            spiketimes.append(spike_t[indices])

        length = np.around(whole_length-stim_end, 1) if post is None else post

    #if burst_detection(spiketimes, whole_length):
    #    print(record)
    #    continue

    stm_id = int(record.split('/')[-1].split('.')[0].split('_')[0])

    bin_edges_pre, ifr_pre = to_ifr(spiketimes, wwidth, wshift, 0, -pre)
    bin_edges_post, ifr_post = to_ifr(spiketimes, wwidth, wshift, length, offset)
    bin_edges = np.concatenate((bin_edges_pre, bin_edges_post), axis=0)
    ifr = np.concatenate((ifr_pre, ifr_post), axis=1).tolist()

    X.append(ifr)
    y.append(stm_id)

X = np.array(X)
y = np.array(y)
print(X.shape)
print(bin_edges)

if outfile is not None:
    np.savez(outfile, bin_edges=bin_edges, X=X, y=y,
        wwidth=wwidth, wshift=wshift)

#import matplotlib.pyplot as plt
#fig, axs = plt.subplots(nrows=2, figsize=(12, 7), sharex=True)
#for ifr in X:
#    axs[0].plot(bin_edges, np.mean(ifr, axis=0), c='k', alpha=0.2)
#axs[1].plot(bin_edges, np.mean(X, axis=(0, 1)), c='k')
#axs[1].set_xlim(bin_edges[0]-wshift, bin_edges[-1])
#plt.show()


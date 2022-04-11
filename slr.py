
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut

parser = argparse.ArgumentParser()
parser.add_argument('infile',
    type=str, help='ifr file (npz)')
parser.add_argument('outfile',
    type=str, help='outfilename (npz)')
parser.add_argument('-s', '--seed',
    type=int, default=2475165645, help='random seed (int)')
parser.add_argument('-l', '--length',
    type=float, default=0.2, help='length (second)')
parser.add_argument('--shuffle',
    type=int, default=None, help='seed number for shuffle (int)')
args = parser.parse_args()

infile = args.infile
outfile = args.outfile
seed = args.seed
length = args.length
shuffle = args.shuffle

X = None
y = None
bin_edges = None
wwidth = None
with np.load(infile, 'r') as data:
    bin_edges = data['bin_edges']
    indices = np.where(bin_edges<=length*1.1)[0]
    bin_edges = bin_edges[indices]

    X = data['X'][:, :, indices]
    #X = data['X']
    y = data['y']
    #print(X.shape)
    #print(y)
    wwidth = data['wwidth']
n_classes = len(set(y))
#X = (X*wwidth).astype(np.int32)
if shuffle is not None:
    np.random.seed(shuffle)
    np.random.shuffle(y)

loo = LeaveOneOut()
n_s = loo.get_n_splits(X)

#Cs = (0.1, 1, 10, 100, 1000, 10000)
Cs = [10]

accs = np.zeros((X.shape[2], len(Cs), n_s), dtype=np.float32)
coefs = np.zeros((X.shape[2], len(Cs), n_s, len(set(y))), dtype=np.int16)
for j, C in enumerate(Cs):
    clf = LogisticRegression(C=C, penalty='l1', solver='liblinear',
        #multi_class='multinomial', max_iter=10000)
        max_iter=100000)
    for i, bin_edge in enumerate(bin_edges):
        #for k, (train_index, test_index) in enumerate(sss.split(X[:, :, i], y)):
        for k, (train_index, test_index) in enumerate(loo.split(X[:, :, i], y)):
            clf.fit(X[train_index, :, i], y[train_index])

            y_pred = clf.predict(X[test_index, :, i])
            acc = accuracy_score(y[test_index], y_pred)
            accs[i, j, k] = acc
            coefs[i, j, k] = np.sum(clf.coef_!=0., axis=1)

if outfile is not None:
    np.savez(outfile, accs=accs, coefs=coefs, Cs=Cs, bin_edges=bin_edges, infile=infile,
        n_classes=n_classes, n_splits=n_s, wwidth=wwidth)


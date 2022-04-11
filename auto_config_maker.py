
import argparse
import json
import collections
import itertools
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('exp_setting_file', type=str)
parser.add_argument('outdir', type=str)
parser.add_argument('--string_format_file',
    default='string_format.json', type=str)
args = parser.parse_args()

exp_setting = None
with open(args.exp_setting_file, 'r') as f:
    decoder = json.JSONDecoder(object_pairs_hook=collections.OrderedDict)
    exp_setting = decoder.decode(f.read())

string_format = None
with open(args.string_format_file, 'r') as f:
    string_format = json.load(f)

keys = []
list_values = []
formats = []
ms = []
ABBs = []
for key, values in exp_setting.items():
    keys.append(key)
    list_values.append(values)
for items in [ string_format[key] for key in keys ]:
    formats.append(items['fmt'])
    ms.append(1/items['base'])
    ABBs.append(items['ABB'])

p = itertools.product(*list_values)
for condition in p:
    fname = 'config'
    options = ''
    for val, fmt, m, ABB, key in zip(condition, formats, ms, ABBs, keys):
        my_round_int = lambda x: int((x*2+1)//2)
        fname += '_'+ABB+fmt%(my_round_int(val*m))
        options += ' --'+key+(' %f'%val if type(val)==float else ' %d'%val)
    fname += '.json'

    subprocess.call('python make_config.py '+args.outdir+'/'+fname+options,
        shell=True)
    print(args.outdir+'/'+fname)


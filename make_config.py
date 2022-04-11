
import argparse
import json

params = None
with open('default_config.json', 'r') as f:
    params = json.load(f)

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
for key, value in params.items():
    parser.add_argument('--%s'%key, type=type(value), default=value)
args = parser.parse_args()

content = '''
{
    "N": %d,
    "inhibitory_ratio": %.2f,

    "tau": %d,
    "ref_period_exp": %d, "ref_period_inh": %d,
    "rate": %.2f,
    "v_th": %d, "v_rest": %d, "v_reset": %d,

    "E_exc": %d, "E_inh": %d,
    "tau_ampa": %d, "tau_gaba": %d,
    "delay_EE": %.2f, "delay_other": %.2f,
    "w0": %.2f,
    "gmax_exc": %.2f, "gmax_inh": %.2f,
    "p_connection": %.2f,

    "U": %.2f, "delta_U": %.2f,
    "tau_rec": %d,

    "beta": %.2f,
    "tau_estdp": %d,
    "delta_estdp": %f,

    "alpha": %.2f,
    "tau_1": %d, "tau_2": %d,
    "delta_istdp": %f
}
'''%(
args.N,
args.inhibitory_ratio,
args.tau,
args.ref_period_exp, args.ref_period_inh,
args.rate,
args.v_th, args.v_rest, args.v_reset,
args.E_exc, args.E_inh,
args.tau_ampa, args.tau_gaba,
args.delay_EE, args.delay_other,
args.w0,
args.gmax_exc, args.gmax_inh,
args.p_connection,
args.U, args.delta_U,
args.tau_rec,
args.beta,
args.tau_estdp,
args.delta_estdp,
args.alpha,
args.tau_1, args.tau_2,
args.delta_istdp
)

with open(args.config, 'w') as f:
    f.write(content)


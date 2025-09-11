import numpy as np

from data_loader import load_network_from_csv, digraph_to_adjacency_matrix
from q_model import SI,SIR
from visualization import plot_line_chart

MAX_TIME = 1
CYCLE_TIME = 0.001
SPREADING_SOURCE = 5


def run_SI(nodes, adj_mtx):
    sp = SI(nodes, adj_mtx, lam=0.5, alpha=1.0)
    init_spreading = np.zeros(len(nodes))
    init_spreading[SPREADING_SOURCE] = 1.0
    return sp.run(init_spreading_state=init_spreading, max_time=MAX_TIME, cycle_time=CYCLE_TIME)

def run_SIR(nodes, adj_mtx):
    sp = SIR(nodes, adj_mtx, lam=0.5, alpha=1.0, delta1=0.5)
    init_spreading = np.zeros(len(nodes)+1)
    init_spreading[SPREADING_SOURCE] = 1.0
    return sp.run(init_spreading_state=init_spreading, max_time=MAX_TIME, cycle_time=CYCLE_TIME)

def run_SIRS(nodes, adj_mtx):
    sp = SIR(nodes, adj_mtx, lam=0.5, alpha=1.0, delta1=0.5, delta2=0.5)
    init_spreading = np.zeros(len(nodes)+1)
    init_spreading[SPREADING_SOURCE] = 1.0
    return sp.run(init_spreading_state=init_spreading, max_time=MAX_TIME, cycle_time=CYCLE_TIME)

if __name__ == "__main__":
    network = load_network_from_csv("data/network.csv")
    adj_mtx, nodes = digraph_to_adjacency_matrix(network)

    for run in [run_SI, run_SIR, run_SIRS]:
        trajectary, distances = run_SIR(nodes, adj_mtx)

    
    plot_line_chart([np.array(range(len(distances['kl'])))], [distances['kl']], labels=['kl distance to the stationary'])
        
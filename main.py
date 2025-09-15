import numpy as np

from data_loader import load_network_from_csv, digraph_to_adjacency_matrix, save_to_json, load_json_file
from q_model import SI,SIR,SIRS
from visualization import plot_line_chart
from utils import get_today, dict_to_single_line_text

MAX_TIME = 1
CYCLE_TIME = 0.001
SPREADING_SOURCE = 5


def init_SI(nodes, adj_mtx):
    return SI(nodes, adj_mtx, lam=0.5, alpha=1.0)
    

def init_SIR(nodes, adj_mtx):
    return SIR(nodes, adj_mtx, lam=0.5, alpha=1.0, delta1=0.5)
    

def init_SIRS(nodes, adj_mtx):
    return SIRS(nodes, adj_mtx, lam=0.5, alpha=1.0, delta1=0.5, delta2=0.001)


if __name__ == "__main__":
    network = load_network_from_csv("data/network.csv")
    adj_mtx, nodes = digraph_to_adjacency_matrix(network)

    sp = SI(nodes, adj_mtx, lam=0.5, alpha=1.0)
    init_spreading = np.zeros(len(sp.states))
    init_spreading[SPREADING_SOURCE] = 1.0
    dynamics, distances = sp.run(init_spreading_state=init_spreading, max_time=MAX_TIME, cycle_time=CYCLE_TIME)

    results = {
        'model': sp.__class__.__name__,
        'params': sp.params,
        'start_state': init_spreading,
        'stationary': sp.stationary,
        'dynamics': dynamics,
        'distances': distances
    }

    print(results)



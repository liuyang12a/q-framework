import numpy as np

from data_loader import load_network_from_csv, digraph_to_adjacency_matrix
from q_model import SI,SIR
from visualization import plot_discrete_distribution_curve

MAX_TIME = 1
CYCLE_TIME = 0.001
SPREADING_SOURCE = 5


def run_SI(nodes, adj_mtx):
    sp = SI(nodes, adj_mtx, lam=0.5, alpha=1.0)
    init_spreading = np.zeros(len(nodes))
    init_spreading[5] = 1.0
    trajectary = sp.run(init_spreading_state=init_spreading, max_time=MAX_TIME, cycle_time=CYCLE_TIME)
    return trajectary

def run_SIR(nodes, adj_mtx):
    sp = SIR(nodes, adj_mtx, lam=0.5, alpha=1.0, delta1=0.5)
    init_spreading = np.zeros(len(nodes)+1)
    init_spreading[5] = 1.0
    trajectary = sp.run(init_spreading_state=init_spreading, max_time=MAX_TIME, cycle_time=CYCLE_TIME)
    return trajectary


if __name__ == "__main__":
    network = load_network_from_csv("data/network.csv")
    adj_mtx, nodes = digraph_to_adjacency_matrix(network)



    trajectary = run_SI(nodes, adj_mtx)
    for i in range(1000):
        if i%100==0:
            plot_discrete_distribution_curve(
                values=np.array(range(len(trajectary[i]))),
                probabilities=trajectary[i],
                title='%d'%i,
                x_label='node id',
                y_label='influenced prob',
                show_points=False
            )
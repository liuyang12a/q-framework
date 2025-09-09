import numpy as np

from data_loader import load_network_from_csv, digraph_to_adjacency_matrix
from q_model import SI
from visualization import plot_matrix_heatmap

if __name__ == "__main__":
    network = load_network_from_csv("data/network.csv")
    adj_mtx, nodes = digraph_to_adjacency_matrix(network)

    sp = SI(nodes, adj_mtx, lam=0.5, alpha=1.0)
    init_spreading = np.zeros(len(nodes))
    print(init_spreading)
    init_spreading[5] = 1.0
    trajectary = sp.run(init_spreading_state=init_spreading, max_time=1, cycle_time=0.001)
    plot_matrix_heatmap(np.array(trajectary).T, figsize=(50,10), vmax=0.007,vmin=0.0045)
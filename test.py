import numpy as np

from data_loader import load_network_from_csv, digraph_to_adjacency_matrix
from q_model import SI,SIR
from visualization import plot_matrix_heatmap, DistributionVisualizer, plot_discrete_distribution_curve
from ctmc import array_to_discrete_distribution

if __name__ == "__main__":
    nodes = [1,2]
    adj_mtx = np.array([[0,1],[1,0]])

    sp = SIR(nodes, adj_mtx, lam=0.5, alpha=1.0, delta1=0.5)
    print(sp.Q)
    init_spreading = np.zeros(len(nodes)+1)
    print(init_spreading)
    init_spreading[2] = 1.0
    sp.set_cycle_stride(1)
    print(sp.transition_matrix)
    state = init_spreading
    for i in range(5):
        next_state = sp.forward(state)
        print(next_state)
        state = next_state
    
    # for i in range(1000):
    #     if i%100==0:
    #         plot_discrete_distribution_curve(
    #             values=np.array(range(len(trajectary[i]))),
    #             probabilities=trajectary[i],
    #             title='%d'%i,
    #             x_label='node id',
    #             y_label='influenced prob',
    #             show_points=False
    #         )
import numpy as np
from tqdm import tqdm

from data_loader import load_network_from_csv, digraph_to_adjacency_matrix, save_to_json, load_json_file
from q_model import SI, SIR, SIRS
from utils import sample_inverse_gamma
from visualization import plot_discrete_distribution_curve

NODE_TYPE = {
    "轻信者": 0,
    "社交表现者": 1,
    "阴谋论者": 2,
    "谨慎分享者": 3,
    "批判思考者": 4
}

INFLUENCE_MATRIX = [
    [1.0, 3.0],
    [3.0, 1.0], 
    [3.0, 3.0],
    [0.5, 0.5],
    [1.5, 0.5]
]

MAX_TIME = 1
CYCLE_TIME = 0.001
SPREADING_SOURCE = 5

if __name__ == "__main__":
    network = load_network_from_csv("data/network.csv")
    adj_mtx, nodes = digraph_to_adjacency_matrix(network)
    # following matrix to spreading matrix
    adj_mtx  = adj_mtx.T
    profiles = load_json_file('data/population_profiles.json')

    person_type = []
    groups = ['gullible', 'performative', 'conspiracist', 'cautious', 'rational']
    person_gorup = {p:[] for p in  groups}
    colors =['orange', 'yellow', 'red', 'blue', 'purple']
    person_color = {p:c for p, c in zip(groups, colors)}
    for p in profiles:
        nt = NODE_TYPE[p['message_propagation_preference']]
        person_type.append(nt)
        person_gorup[groups[nt]].append(len(person_type)-1)


    
    
    for source in nodes:
        for target in nodes:
            # adj_mtx[source][target] *= max(0.1, 
            #                                sample_inverse_gamma(shape=1.0, 
            #                                                     scale=INFLUENCE_MATRIX[person_type[source]][0]*INFLUENCE_MATRIX[person_type[target]][1], 
            #                                                     size=1))
            adj_mtx[source][target] = INFLUENCE_MATRIX[person_type[source]][0]*INFLUENCE_MATRIX[person_type[target]][1]
    # sp = SI(nodes, adj_mtx, lam=None, alpha=1.0)
    sp = SIR(nodes, adj_mtx, lam=None, alpha=1.0, delta1=1.0)
    # sp = SIRS(nodes, adj_mtx, lam=None, alpha =1.0, delta1=1.0, delta2=1.0)
    person_gorup['#0'] = [len(nodes)]
    person_color['#0'] = 'green'
    init_spreading = np.zeros(len(sp.states))
    init_spreading[SPREADING_SOURCE] = 1.0
    dynamics, distances = sp.run(init_spreading_state=init_spreading, max_time=MAX_TIME, cycle_time=CYCLE_TIME)
    results = {
        'model': sp.__class__.__name__,
        'params': sp.params,
        'start_state': init_spreading,
        'stationary': sp.stationary,
        # 'dynamics': dynamics,
        'distances': distances
    }
    # print(results['stationary'])
    plot_discrete_distribution_curve(values=range(len(results['stationary'])), 
                                     probabilities=results['stationary'], 
                                     title='stationary', 
                                     show_points=True,
                                     color_group=(person_gorup, person_color)
                                     )
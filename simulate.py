import numpy as np
from tqdm import tqdm

from data_loader import load_network_from_csv, digraph_to_adjacency_matrix, save_to_json, load_json_file
from q_model import SI,SIR,SIRS
from visualization import plot_line_chart, plot_discrete_distribution_curve
from utils import get_today, dict_to_single_line_text

MAX_TIME = 1
CYCLE_TIME = 0.001
SPREADING_SOURCE = 5


def sim_SI(nodes, adj_mtx):
    total_point = 9
    with tqdm(total=total_point) as pbar:
        for lam in np.linspace(0.1,0.9,total_point):

            sp = SI(nodes, adj_mtx, lam=lam, alpha=1.0)
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

            # plot_discrete_distribution_curve(values=range(len(results['stationary'])), 
            #                                  probabilities=results['stationary'], 
            #                                  title='SI-lambda=%.2f'%(results['params']['lambda']))
            save_to_json(results, "cache/SI/%s/lam=%.2f.json"%(get_today(),lam))
            pbar.update(1)
    

def sim_SIR(nodes, adj_mtx):
    total_point = 9
    with tqdm(total=total_point*total_point) as pbar:
        for lam in np.linspace(0.1,0.9,total_point):
            for delta1 in np.linspace(0.1,0.9,total_point):
                sp = SIR(nodes, adj_mtx, lam=lam, alpha=1.0, delta1=delta1)
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

                save_to_json(results, "cache/SIR/%s/lam=%.2f-delta1=%.2f.json"%(get_today(),lam,delta1))
                pbar.update(1)

    

def sim_SIRS(nodes, adj_mtx):
    total_point = 9
    lam = 0.5
    with tqdm(total=total_point*total_point) as pbar:
        for delta1 in np.linspace(0.9,0.1,total_point):
            for delta2 in np.linspace(0.1, 0.9, total_point):

                sp = SIRS(nodes, adj_mtx, lam=lam, alpha=1.0, delta1=delta1, delta2=delta2)
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
                # plot_discrete_distribution_curve(values=range(len(results['stationary'])), probabilities=results['stationary'], title='SIRS-lambda=%.2f,delta1=%.2f,delta2=%.2f'%(results['params']['lambda'],results['params']['delta1'],results['params']['delta2']))
                
                save_to_json(results, "cache/SIRS/%s/lam=%.1f-delta1=%.1f-delta2=%.1f.json"%(get_today(),lam,delta1,delta2))
                pbar.update(1)


if __name__ == "__main__":
    network = load_network_from_csv("data/network.csv")
    adj_mtx, nodes = digraph_to_adjacency_matrix(network)
    # following matrix to spreading matrix
    adj_mtx  = adj_mtx.T

    sim_SI(nodes=nodes, adj_mtx=adj_mtx)
    sim_SIR(nodes=nodes, adj_mtx=adj_mtx)
    sim_SIRS(nodes=nodes, adj_mtx=adj_mtx)


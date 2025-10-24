import random
from data_loader import load_json_file
from visualization import plot_line_chart, plot_discrete_distribution_curve
from utils import traverse_files, trans_list2str

import numpy as np
import re
import os


def draw_SI(dir, y_label='kl', title='Distance to the stationary'):
    lams = ['0.2', '0.5', '0.8']

    def filter(s):
        if re.search(r"lam=(%s)"%(trans_list2str(lams,'|')), s) is not None:
            return True
        
    files = traverse_files(dir, filter_func=filter)
    trajectorys = []
    for file in files:
        print(file)
        result = load_json_file(file)
        trajectorys.append(result['distances'][y_label])
    
    plot_line_chart(
        x_data=[range(len(tra)) for tra in trajectorys],
        y_data=trajectorys,
        labels=['λ=%s'%(s) for s in lams],
        x_label='time_steps',
        y_label=y_label + '-distance to stationary',
        title=title,
        log_x=True,
        log_y=True
    )

def draw_SIR(dir, y_label='kl', title='Distance to the stationary'):
    lams = ['0.5']
    d1s = ['0.2', '0.5', '0.8']

    def filter(s):
        if re.search(r"lam=(%s).*delta1=(%s)"%(trans_list2str(lams,'|'), trans_list2str(d1s, '|')), s) is not None:
            return True

    files = traverse_files(dir, filter_func=filter)
    labels = []
    trajectorys = []
    for file in files:
        print(file)
        result = load_json_file(file)
        trajectorys.append(result['distances'][y_label])
        labels.append('λ=%.1f,δ1=%.1f'%(result['params']['lambda'], result['params']['delta1']))
    plot_line_chart(
        x_data=[range(len(tra)) for tra in trajectorys],
        y_data=trajectorys,
        labels=labels,
        x_label='time_steps',
        y_label=y_label + '-distance to stationary',
        title=title,
        log_x=True,
        log_y=True
    )


def draw_SIRS(dir, y_label='kl', title='Distance to the stationary'):
    d1s = ['0.1', '0.5', '0.9']
    d2s = ['0.1', '0.5', '0.9']

    def filter(s):
        if re.search(r"delta1=(%s).*delta2=(%s)"%(trans_list2str(d1s,'|'), trans_list2str(d2s, '|')), s) is not None:
            return True
        
    files = traverse_files(dir, filter_func=filter)
    labels = []
    trajectorys = []
    for file in files:
        print(file)
        result = load_json_file(file)
        trajectorys.append(result['distances'][y_label])
        labels.append('δ1=%.1f,δ2=%.1f'%(result['params']['delta1'], result['params']['delta2']))
    plot_line_chart(
        x_data=[range(len(tra)) for tra in trajectorys],
        y_data=trajectorys,
        labels=labels,
        x_label='time_steps',
        y_label=y_label + '-distance to stationary',
        title=title,
        log_x=True,
        log_y=True
    )

def draw_stationary(file):
    result = load_json_file(file)
    plot_discrete_distribution_curve(values=range(len(result['stationary'])), probabilities=result['stationary'],fixed_lim=False,title="", show_points=False)

def draw_heter(dir, y_label='cosine', title=''):
    files = traverse_files(dir, filter_func=lambda x:x)
    labels = []
    trajectorys = []
    for file in files:
        print(file)
        result = load_json_file(file)
        trajectorys.append(result['distances'][y_label])
        labels.append(os.path.splitext(os.path.basename(file))[0])
    plot_line_chart(
        x_data=[range(len(tra)) for tra in trajectorys],
        y_data=trajectorys,
        labels=labels,
        x_label='time_steps',
        y_label=y_label + '-distance to stationary',
        title=title,
        log_x=True,
        log_y=True
    )

def draw_point_dynamics(file):
    groups = ['gullible', 'performative', 'conspiracist', 'cautious', 'rational']
    points =  random.sample(range(200), k=6)
    # points = [5]
    print(points)
    result = load_json_file(file)
    dynamics =  result['dynamics']
    person_type = result['person_type']
    dynamics = np.vstack(dynamics)
    labels = []
    trajectorys = []
    for p in points:
        labels.append('node_%d(%s)'%(p, groups[person_type[p]]))
        traj = dynamics[:,p]
        trajectorys.append(traj)
    plot_line_chart(
        x_data=[range(len(tra)) for tra in trajectorys],
        y_data=trajectorys,
        labels=labels,
        x_label='time_steps',
        y_label='relative influenced intensity',
        title="",
        log_x=False,
        log_y=False,
        linewidth=2,
        legend_ncol=1
    )

def draw_state(file):
    result = load_json_file(file)
    dynamics = result['dynamics']
    
    for i in range(10):
        plot_discrete_distribution_curve(values=range(len(dynamics[0])), probabilities=dynamics[i],fixed_lim=False,title="", show_points=False)


    

if __name__ == "__main__":

    # draw_SI('cache/SI/2025-10-17/', y_label='kl', title="")
    # draw_SIR('cache/SIR/2025-10-17/', y_label='kl', title="")
    # draw_SIRS('cache/SIRS/2025-10-17/', y_label='kl', title="")
    # draw_stationary('cache/SI/2025-10-17/lam=0.50.json')
    # draw_stationary('cache/SIR/2025-10-17/lam=0.10-delta1=0.50.json')
    # draw_stationary('cache/SIRS/2025-10-17/lam=0.5-delta1=0.9-delta2=0.1.json')

    # draw_heter('cache/SI-heter/2025-10-22/')
    draw_point_dynamics('cache/SI-heter/2025-10-22/normal.json')
    # draw_state('cache/SI-heter/2025-10-22/normal.json')

    p1 = [80,84,125,118,185]

    p2 = [11,14,28,132,138]

    p3 = [29, 3, 184, 89, 150]

    p4 = [47, 45, 6, 98, 139]

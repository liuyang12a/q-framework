from data_loader import load_json_file
from visualization import plot_line_chart, plot_discrete_distribution_curve
from utils import traverse_files, trans_list2str

import numpy as np
import re


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
    plot_discrete_distribution_curve(values=range(len(result['stationary'])), probabilities=result['stationary'],fixed_lim=False)



if __name__ == "__main__":

    draw_SI('cache/SI/2025-10-17/', y_label='cosine', title="SI")
    draw_SIR('cache/SIR/2025-10-17/', y_label='cosine', title="SIR")
    draw_SIRS('cache/SIRS/2025-10-17/', y_label='cosine', title="SIRS")
    # draw_stationary('cache/SI/2025-10-14/lam=0.50.json')
    # draw_stationary('cache/SIR/2025-10-14/lam=0.10-delta1=0.50.json')
    # draw_stationary('cache/SIRS/2025-10-14/lam=0.5-delta1=0.9-delta2=0.1.json')

import os

import time
from datetime import datetime
import sys
import numpy as np
import pandas

import argparse
import pandas as pd
from experiments.vkt_experiment.analysis_utilities import visualize_average_probability
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import matplotlib
from matplotlib.cm import get_cmap
cmap = get_cmap('tab10')
font = {'weight' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)


def stitch_label_prob_dicts(a, b):
    tmp = {}
    for _label_key in a.keys():
        tmp[_label_key] = {}
        for _metric in a[_label_key].keys():
            tmp[_label_key][_metric] = np.concatenate([a[_label_key][_metric],
                                                       b[_label_key][_metric]], axis=-1)
            print(a[_label_key][_metric].shape, b[_label_key][_metric].shape, tmp[_label_key][_metric].shape)


    return tmp


dataset_to_label_names = {
    'butterflies': ['Cabbage White', 'Monarch', 'Queen', 'Red Admiral', 'Viceroy'],
    'oct': ['DME', 'Drusen', 'Normal'],
    'greebles': ['Agara', 'Bari', 'Cooka']
}

script_name_to_paper_names = {
    'runner_baseline': 'Static Model',
    # 'runner_timestep_baseline': 'Time-Sensitive Model',
    # 'runner_direct_response_model1v3': 'Direct Model',
    'runner_classifier_model1v2': 'Classifier Model',
}

tag_name = 'experiment1'
path = f'../experiment_analysis_output/{tag_name}/'
os.makedirs(path, exist_ok=True)

with open(path + 'metric_data_dict_average_prob.pkl', 'rb') as file:
    metric_data_dict = pkl.load(file)

dataset_indexed_mdd = {}

for ki, key in enumerate(metric_data_dict):

    entry = metric_data_dict[key]
    dataset = entry['dataset']
    script_name = entry['script_name']

    print(script_name, dataset)
    if '0' not in dataset:
        continue

    if dataset not in dataset_indexed_mdd:
        dataset_indexed_mdd[dataset] = {}
    dataset_indexed_mdd[dataset][script_name] = entry

for dataset in dataset_indexed_mdd:

    entry = dataset_indexed_mdd[dataset]
    if 'butterflies' in dataset.lower():
        labels = dataset_to_label_names['butterflies']
    elif 'oct' in dataset.lower():
        labels = dataset_to_label_names['oct']
    else:
        labels = dataset_to_label_names['greebles']

    fig, axes = None, None
    ci = 1
    for si, script_name in enumerate(entry):
        if script_name not in script_name_to_paper_names:
            continue
        model_name = script_name_to_paper_names[script_name]
        label_prob_dict_train = entry[script_name]['label_probability_dict_train']
        label_prob_dict_test = entry[script_name]['label_probability_dict_test']
        label_prob_dict = stitch_label_prob_dicts(label_prob_dict_train, label_prob_dict_test)
        if fig is None:
            fig, axes = plt.subplots(1, len(label_prob_dict_train), squeeze=False, constrained_layout=True)
            axes = axes[0]
            fig.set_size_inches(15, 3.27)

        legend_labels = [model_name] * len(label_prob_dict_train)
        plot_params = visualize_average_probability.plot_label_prob_dict(axes, label_prob_dict, color=cmap(ci),
                                                           mode='per_timestep', legend_labels=legend_labels)
        ci += 1
    for axi, ax in enumerate(axes):
        # ax.set_title(labels[axi])
        ax.axvline(30, color='red')
        ax.set_xlim((-2.2, 46.2))
        ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
        if axi > 0:
            ax.set_yticklabels([])

        # print(ax.get_ylim())
    axes[0].set_ylabel('Average Probability')
    midpt_ind = len(axes) // 2
    axes[midpt_ind].set_xlabel('Time Step')
    axes[midpt_ind].legend(fontsize=10)

    path = f'../figures/{tag_name}/model_average_probabilities_main/'
    os.makedirs(f'{path}', exist_ok=True)
    plt.savefig(f'{path}/{dataset}_average_probabilities.pdf')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import numpy as np
import plotting
import os
import experiments.vkt_experiment.experiment_utilities as eut
import argparse
import sys
from experiments.vkt_experiment.analysis_utilities import misc_functions as f
from experiments.vkt_experiment.analysis_utilities import test_sequence_forward_functions as mff
import umap
import torch
import pickle as pkl
import matplotlib
font = {'weight' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)

experiment_data_names = ['butterflies']
dataset_list = [f'butterflies_fold1']

script_names = [
    'runner_classifier_model1v2',
    # 'runner_direct_response_model1v3'
]
tag_name = 'experiment1'
experiment_group_name = eut.experiment_group_name
dataset_to_label_names = {
    'butterflies': ['Cabbage White', 'Monarch', 'Queen', 'Red Admiral', 'Viceroy'],
    'oct': ['DME', 'Drusen', 'Normal'],
    'greebles': ['Agara', 'Bari', 'Cooka']
}

script_name_to_paper_names = {
    'runner_baseline': 'Static Model',
    'runner_timestep_baseline': 'Static Time',
    'runner_direct_response_model1v3': 'Direct',
    'runner_classifier_model1v2': 'Classifier Model',
}

# dict of dicts stores data that is necessary for plotting [indexed primarily by script name (model type)]
metric_data_dict = {}
# dict of dicts stores the summary statistic from a metric for a table [indexed primarily by script name (model type)]
metric_summary_dict = {}
count = 0
for d, dataset in enumerate(dataset_list):
    experiment_data_name = experiment_data_names[d]
    experiment_data_specifier = dataset
    # if d == 0 or d == 1:
    #     continue
    for script_name in script_names:
        _metric_summary_dict = {'script_name': script_name, 'dataset': dataset}
        _metric_data_dict = {'script_name': script_name, 'dataset': dataset}

        batch_size = 16
        if 'sld' in dataset:
            batch_size = 32

        print('-' * 89)
        print(experiment_data_specifier, script_name)
        epochs = 250
        model_name = script_name.replace('runner_', '')

        experiment_name = model_name
        experiment_save_specifier = f'/{tag_name}/model={model_name}_dataset={dataset.replace("/", "_")}'
        filename = 'best.pt'
        model_dict = eut.load_model(experiment_group_name, experiment_name, experiment_save_specifier, filename)

        model = model_dict['model']

        # LOAD DATA AND GENERALLY FORMAT FOR MODEL
        # ----------------------------------------------------------------------------------------------------------- #
        if 'sld' not in experiment_data_specifier:
            data_folder = os.path.join(experiment_data_name, experiment_data_specifier + '_train_test')
        else:
            data_folder = os.path.join(experiment_data_name, experiment_data_specifier)
        learner_data, experiment_vars, training_data = eut.load_learner_data(data_folder, get_training_data=True,
                                                                             device='cpu')
        evars = argparse.Namespace(**experiment_vars['input_params'])
        mask_dict = experiment_vars['mask_dict']
        split_dict = experiment_vars['split_dict']

        device = 'cpu'
        # LOAD MODEL
        model = model.requires_grad_(False).to(device)
        # SEPARATE INTO TRAIN AND TEST SEQ IF NECESSARY, ALSO FORMAT DATA FOR THE MODEL BEING EVALUATED
        # ----------------------------------------------------------------------------------------------------------- #
        split = 'test'
        if hasattr(evars, 'train_sequence_length'):
            evars.train_sequence_length = int(evars.train_sequence_length)  # saved as tensor unfortunately
            data_dict_per_seq = dict(
                train_seq=f.prep_data_for_metrics(model, learner_data[:, :evars.train_sequence_length], mask_dict,
                                                  split_dict, training_data, evars),
                test_seq=f.prep_data_for_metrics(model, learner_data[:, evars.train_sequence_length:], mask_dict,
                                                 split_dict, training_data, evars)
            )
        else:
            data_dict_per_seq = dict(
                train_seq=f.prep_data_for_metrics(model, learner_data[:, :evars.sequence_length], mask_dict,
                                                  split_dict, training_data, evars),
            )

        # EXECUTE MODEL ON FORMATTED DATA
        # ----------------------------------------------------------------------------------------------------------- #
        data_dict_per_seq['train_seq']['out'] = model(**data_dict_per_seq['train_seq']['formatted_data'][split])

        test_seq_exists = 'test_seq' in data_dict_per_seq
        if test_seq_exists:
            data_dict_per_seq['test_seq']['out'] = mff.execute_test_seq_forward(model,
                                                                                data_dict_per_seq['test_seq'][
                                                                                    'formatted_data'][split],
                                                                                data_dict_per_seq['train_seq']['out'])
            test_seq = argparse.Namespace(**data_dict_per_seq['test_seq'])

        train_seq = argparse.Namespace(**data_dict_per_seq['train_seq'])

        # VISUALIZE LSTM HIDDEN DIMENSION
        # ----------------------------------------------------------------------------------------------------------- #

        hidden_states = train_seq.out['hidden_states'][:, [-1]].permute(2, 0, 1, 3)
        cell_states = train_seq.out['cell_states'][:, [-1]].permute(2, 0, 1, 3)
        orig_shape = hidden_states.shape
        num_learners, seq_len, num_layers, dim = orig_shape
        hidden_states = hidden_states.flatten(0, -2)
        cell_states = cell_states.flatten(0, -2)

        y = train_seq.formatted_data[split]['teaching_signal_t'].argmax(-1)
        r = train_seq.formatted_data[split]['responses_t'].argmax(-1).squeeze(-1)
        c = y == r
        c = c.type(torch.float)
        print(c.mean(-1))
        time_step = torch.arange(30).reshape(1, -1, 1, 1).repeat(orig_shape[0], 1, 1, 1) / 30

        path_to_reductions = f'../figures/{tag_name}/experiment_lstm_reductions/{experiment_save_specifier}/'
        os.makedirs(path_to_reductions, exist_ok=True)
        if os.path.isfile(path_to_reductions + 'reductions.pkl'):
            with open(f'{path_to_reductions}/reductions.pkl', 'rb') as file:
                d = pkl.load(file)
                hid_2d = d['hid_2d']
                cell_2d = d['cell_2d']
                feats = d['feats']
        else:
            with open(f'{path_to_reductions}/reductions.pkl', 'wb') as file:
                reducer = umap.UMAP()
                hid_2d = reducer.fit_transform(hidden_states).reshape(*orig_shape[:-1], 2)
                cell_2d = reducer.fit_transform(cell_states).reshape(*orig_shape[:-1], 2)
                feats = model.feature_extraction(training_data['data'])
                d = dict(hid_2d=hid_2d, cell_2d=cell_2d, feats=feats)
                pkl.dump(d, file)

        labels = training_data['labels']
        hyperplanes = train_seq.out['hyperplanes']
        probs = torch.softmax(
            torch.einsum('bscd, nd -> bsnc', hyperplanes, feats) + train_seq.out['bias'].repeat(1, 1, feats.shape[0],
                                                                                                1), dim=-1)
        hyperplane_ap = torch.zeros(probs.shape[0], 30, 5)
        for li, label in enumerate(labels.unique()):
            mask = label == labels
            hyperplane_ap[:, :, li] = probs[:, :, mask, li].mean(2)

        path = f'../figures/{tag_name}/lstm_figure/'
        os.makedirs(path, exist_ok=True)
        learner_labels = ['A', 'B']
        label_names = dataset_to_label_names['butterflies']
        fig, axes = plt.subplots(2, 5, squeeze=False, constrained_layout=True)
        fig.set_size_inches(15, 5)
        for axi, ax in enumerate(axes[0]):
            axes[0, axi].scatter(hid_2d[:, :, 0, 0], hid_2d[:, :, 0, 1], cmap='plasma',
                                 c=hyperplane_ap[:, :, axi], zorder=10, vmin=0, vmax=1)
            axes[1, axi].scatter(cell_2d[:, :, 0, 0], cell_2d[:, :, 0, 1], cmap='plasma',
                                 c=hyperplane_ap[:, :, axi], zorder=10, vmin=0, vmax=1)
            axes[0, axi].set_title(label_names[axi])
            if axi == 0:
                axes[0, 0].set_ylabel('Hidden State')
                axes[1, 0].set_ylabel('Cell State')

            if axi != 0:
                axes[0, axi].set_xticklabels([])
                axes[1, axi].set_xticklabels([])
                axes[0, axi].set_yticklabels([])
                axes[1, axi].set_yticklabels([])

        plt.savefig(path + 'states_by_class.pdf')
        fig_learner_responses, axes_lr = plt.subplots(1, 1, constrained_layout=True)
        fig_learner_responses.set_size_inches(12, 3)
        loi = [0, 1]
        for ki, k in enumerate(loi):

            cmask = (y[k] == r[k])
            if ki == 0:
                axes_lr.scatter(torch.where(cmask)[0], np.ones(y[k, cmask].shape[0]) * ki, c='green', label='correct')
                axes_lr.scatter(torch.where(~cmask)[0], np.ones(y[k, ~cmask].shape[0]) * ki, c='red', label='incorrect')
            else:
                axes_lr.scatter(torch.where(cmask)[0], np.ones(y[k, cmask].shape[0]) * ki, c='green')
                axes_lr.scatter(torch.where(~cmask)[0], np.ones(y[k, ~cmask].shape[0]) * ki, c='red')

            fig, axes = plt.subplots(2, 1, squeeze=False, constrained_layout=True)
            fig.set_size_inches(6, 8)
            plt.suptitle(f'Learner {learner_labels[ki]}')

            axes[0, 0].scatter(hid_2d[:, :, 0, 0], hid_2d[:, :, 0, 1], cmap='coolwarm', c=time_step[:, :],
                               zorder=10)
            axes[1, 0].scatter(cell_2d[:, :, 0, 0], cell_2d[:, :, 0, 1], cmap='coolwarm', c=time_step[:, :],
                               zorder=10)

            segments_hid = np.concatenate([hid_2d[k, :][:-1], hid_2d[k, :][1:]], axis=1)
            segments_cell = np.concatenate([cell_2d[k, :][:-1], cell_2d[k, :][1:]], axis=1)
            norm = plt.Normalize(0, seq_len)
            lc = LineCollection(segments_hid, cmap='coolwarm', norm=norm)
            lc_cell = LineCollection(segments_cell, cmap='coolwarm', norm=norm, zorder=20)
            lc.set_array(np.arange(seq_len))
            lc_cell.set_array(np.arange(seq_len))
            axes[0, 0].add_collection(lc)
            axes[1, 0].add_collection(lc_cell)
            fig.savefig(path + f'learner{k}_trajectory.pdf')

        axes_lr.legend()
        axes_lr.set_ylim([-1, 2])
        axes_lr.set_yticks([0, 1])
        axes_lr.set_yticklabels(learner_labels)
        axes_lr.set_xlabel('Time Step (Training)')
        axes_lr.set_ylabel('Learner')
        fig_learner_responses.savefig(path + '/learner_responses.pdf')

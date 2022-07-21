import matplotlib.pyplot as plt

import numpy as np
import os
import experiments.vkt_experiment.experiment_utilities as eut
import argparse
import sys
from experiments.vkt_experiment.analysis_utilities import misc_functions as f
from experiments.vkt_experiment.analysis_utilities import test_sequence_forward_functions as mff

import torch
import pickle as pkl
from sklearn.decomposition import PCA
import matplotlib
from matplotlib.cm import get_cmap

font = {'weight': 'normal',
        'size': 16}

matplotlib.rc('font', **font)

experiment_data_names = ['butterflies']
dataset_list = [f'butterflies_fold1']

script_names = [
    'runner_classifier_model1v2',
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

        # VISUALIZE FEATURE SPACE
        # ----------------------------------------------------------------------------------------------------------- #

        feats = model.feature_extraction(training_data['data'])
        gt = train_seq.formatted_data[split]['teaching_signal_t'].argmax(-1)
        hyperplanes = train_seq.out['hyperplanes']
        subspaces = torch.einsum('bscd, nd -> bsncd', hyperplanes,
                                 feats)  # + train_seq.out['bias'].repeat(1, 1, feats.shape[0], 1)
        subspaces = torch.cat([subspaces, train_seq.out['bias'].repeat(1, 1, feats.shape[0], 1).unsqueeze(-1)], dim=-1)
        feats = torch.cat([feats, torch.zeros(size=(feats.shape[0], 1))], dim=-1)
        predicted_labels = subspaces.sum(-1).argmax(-1)
        hyperplanes_w_bias = torch.cat([hyperplanes, train_seq.out['bias'].permute(0, 1, 3, 2)], dim=-1)

        path_to_reductions = f'../figures/experiment_feature_space/{experiment_save_specifier}/'
        os.makedirs(path_to_reductions, exist_ok=True)
        if False and os.path.isfile(path_to_reductions + 'reductions.pkl'):
            with open(f'{path_to_reductions}/feature_space.pkl', 'rb') as file:
                d = pkl.load(file)
                feats = d['feats']
                feats_2d = d['feats_2d']
                labels = d['labels']
                subspace_2d = d['subspace_2d']
                hyperplanes_2d = d['hyperplanes_2d']
                predicted_labels = d['predicted_labels']
                subspaces = d['subspaces']
        else:
            with open(f'{path_to_reductions}/feature_space.pkl', 'wb') as file:
                labels = training_data['labels']
                subsamp_images = np.random.choice(np.arange(0, feats.shape[0]), size=500, replace=False)
                feats = feats[subsamp_images]
                labels = labels[subsamp_images]
                predicted_labels = predicted_labels[:, :, subsamp_images]
                subspaces_reduced = subspaces[:, :, subsamp_images]

                reducer = PCA(2)
                reducer.fit(subspaces_reduced.flatten(0, -2))
                print('fitted, applying...')
                subspace_2d = reducer.transform(subspaces_reduced.flatten(0, -2)).reshape(*subspaces_reduced.shape[:-1],
                                                                                          -1)
                hyperplanes_2d = reducer.transform(hyperplanes_w_bias.flatten(0, -2)).reshape(
                    *hyperplanes_w_bias.shape[:-1], -1)
                feats_2d = reducer.transform(feats.flatten(0, -2)).reshape(*feats.shape[:-1], -1)
                d = dict(feats=feats, labels=labels, subspace_2d=subspace_2d,
                         hyperplanes_2d=hyperplanes_2d, feats_2d=feats_2d,
                         predicted_labels=predicted_labels, subspaces=subspaces)
                pkl.dump(d, file)

        path = f'../figures/{tag_name}/feature_space/'
        os.makedirs(path, exist_ok=True)
        fig, axes = plt.subplots(1, 1, constrained_layout=True)
        fig.set_size_inches(10, 10)
        un_labels = torch.unique(labels)
        for li, label in enumerate(un_labels):
            mask = label == labels
            axes.scatter(feats_2d[mask, 0], feats_2d[mask, 1])
        axes.set_xlabel('PCA Dim 1')
        axes.set_ylabel('PCA Dim 2')
        plt.savefig(path + 'feature_space.pdf')

        cmap = get_cmap('tab10')
        ts = [0, 5, 15, 25, 29]
        label_names = dataset_to_label_names['butterflies']
        for k in [13]:
            fig, axes = plt.subplots(len(ts), 1, squeeze=False, constrained_layout=True)
            axes = axes[:, 0]
            fig.set_size_inches(10, 10)
            plt.suptitle('Learner X')
            for si, s in enumerate(ts):
                axes[si].set_ylabel(f'Time Step {s}')
                lcd = {}
                for ci in range(5):
                    mask = labels > -1
                    if ci == gt[k, s]:
                        axes[si].scatter(subspace_2d[k, s, :, ci, 0][mask][:-1], subspace_2d[k, s, :, ci, 1][mask][:-1], color=cmap(ci))
                        axes[si].scatter(subspace_2d[k, s, :, ci, 0][mask][-1], subspace_2d[k, s, :, ci, 1][mask][-1],
                                         marker='*', label=label_names[ci], color=cmap(ci))
                    else:
                        if si > 0:
                            axes[si].scatter(subspace_2d[k, s, :, ci, 0][mask], subspace_2d[k, s, :, ci, 1][mask], color=cmap(ci))
                        else:
                            axes[si].scatter(subspace_2d[k, s, :, ci, 0][mask], subspace_2d[k, s, :, ci, 1][mask],
                                            label=label_names[ci], color=cmap(ci))
                    lcd[ci] = cmap(ci)
                # x_bounds = [subspace_2d[k, s, :, :, 0].min(), subspace_2d[k, s, :, :, 0].max()]
                axes[si].legend(fontsize=12, loc='upper left')
            axes[-1].set_xlabel('PCA Dim 1')
            plt.savefig(path + 'subspace.pdf')

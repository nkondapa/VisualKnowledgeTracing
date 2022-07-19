import os

import time
from datetime import datetime
import sys
import numpy as np

from experiments.vkt_experiment import experiment_utilities as eut
import argparse
from experiments.vkt_experiment.analysis_utilities import precision_recall_utilities as pru
from experiments.vkt_experiment.analysis_utilities import misc_functions as f
from experiments.vkt_experiment.analysis_utilities import test_sequence_forward_functions as mff
from experiments.vkt_experiment.analysis_utilities import class_conditioned_average_probability_functions as mac
from experiments.vkt_experiment.analysis_utilities import \
    class_conditioned_average_probability_functions_test_sequence as mac_test
from experiments.vkt_experiment.analysis_utilities import training_statistics as ts
from experiments.vkt_experiment.analysis_utilities import visualize_average_probability
import matplotlib.pyplot as plt
import pickle as pkl

'''
GOAL Compare models on the selected datasets with 5-fold cross validation
Datasets
 - butterflies (human)
 - greebles
 - butterflies (synthetic)

'''

start_time = time.time()
dt_object = datetime.fromtimestamp(start_time)
rng = np.random.default_rng(367354245)
seed = rng.integers(0, 93423501230)
tag_name = sys.argv[0].split('/')[-1].replace('_precision_recall.py', '')

print(os.path.abspath(sys.argv[0]))
kfold = 5
experiment_group_name = eut.experiment_group_name
experiment_data_names = []
dataset_list = []
for fi in range(kfold):
    experiment_data_names.extend(['butterflies', 'greebles', 'oct'])
    dataset_list.extend(
        [f'butterflies_fold{fi}', f'greebles_fold{fi}', f'OCT_fold{fi}']
    )

script_names = [
    'runner_baseline',
    'runner_timestep_baseline',
    'runner_gt_baseline',
    'runner_dkt_translation_model',
    'runner_classifier_model1v1',
    'runner_classifier_model1v2',
    'runner_classifier_model1v3',
    'runner_direct_response_model1v1',
    'runner_direct_response_model1v2',
    'runner_direct_response_model1v3',
    'runner_direct_response_transformer',
    'runner_prototype_baseline',
    'runner_exemplar_baseline',
]

# dict of analysis to compute, along with their parameters
metrics = {'precision_recall': {}}
# dict of dicts stores data that is necessary for plotting [indexed primarily by script name (model type)]
metric_data_dict = {}
# dict of dicts stores the summary statistic from a metric for a table [indexed primarily by script name (model type)]
metric_summary_dict = {}

information_level = 'argmax'
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
        experiment_save_specifier = f'/{tag_name}/model={model_name}_dataset={dataset}'
        filename = 'best.pt'
        model_dict = eut.load_model(experiment_group_name, experiment_name, experiment_save_specifier, filename)

        if 'training_statistics' in metrics:
            ts_dict = ts.pull_training_data_from_model_dict(experiment_group_name, experiment_name,
                                                            experiment_save_specifier)
            metric_data_dict['training_statistics'] = ts_dict
            metric_summary_dict['training_statistics'] = dict(test_loss=ts_dict['test_loss'],
                                                              test_acc=ts_dict['test_acc'])
            fig, axes = plt.subplots(1, 2)
            ts.plot_curves(axes[0], ts_dict, {'metric': 'loss', 'splits': ['train', 'val']})
            ts.plot_curves(axes[1], ts_dict, {'metric': 'acc', 'splits': ['train', 'val']})
            plt.suptitle(_metric_summary_dict)
            plt.show()

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
            _metric_summary_dict['num_correct_test'] = test_seq.formatted_data[split]['num_correct_per_learner']
            _metric_summary_dict['num_correct_per_class_test'] = test_seq.formatted_data[split]['num_correct_per_learner_per_class']
            _metric_data_dict['num_correct_test'] = test_seq.formatted_data[split]['num_correct_per_learner']
            _metric_data_dict['num_correct_per_class_test'] = test_seq.formatted_data[split]['num_correct_per_learner_per_class']

        train_seq = argparse.Namespace(**data_dict_per_seq['train_seq'])

        ###### START COMPUTING METRICS #######

        # COMPUTE PRECISION RECALL
        # ----------------------------------------------------------------------------------------------------------- #
        if 'precision_recall' in metrics:
            mask = None
            if 'test_lower_threshold' in metrics['precision_recall']:
                mask = test_seq.formatted_data[split]['num_correct_per_learner'] < 8
                num_learner_after_threshold = mask.sum()
                _metric_summary_dict['num_learner_after_threshold'] = num_learner_after_threshold.item()
                print(f'Num Learners after mask : {num_learner_after_threshold}/{len(mask)}')

            pr_dict = pru.compute_pr(train_seq.out, train_seq.formatted_data[split], evars, subset_mask=mask)
            _metric_data_dict['pr_data'] = pr_dict
            _metric_summary_dict['train_seq_AP'] = pr_dict['average_precision']
            print(pr_dict['average_precision'])
            if test_seq_exists:

                pr_dict = pru.compute_pr(test_seq.out, test_seq.formatted_data[split], evars, subset_mask=mask)
                _metric_data_dict['test_pr_data'] = pr_dict
                _metric_summary_dict['test_seq_AP'] = pr_dict['average_precision']
                print(pr_dict['average_precision'])

        # COMPUTE AVERAGE PROBABILITY
        # ----------------------------------------------------------------------------------------------------------- #
        if 'average_prob' in metrics:
            print('computed')
            ap_dict_train, subsample_indices = mac.execute_class_conditioned_probability(model,
                                                                                         train_seq.formatted_data[
                                                                                             split], training_data,
                                                                                         train_seq.out,
                                                                                         subsample_fraction=0.1)
            ap_dict_test = mac_test.execute_class_conditioned_probability(model, test_seq.formatted_data[split],
                                                                          training_data,
                                                                          train_seq.out,
                                                                          subsample_indices=subsample_indices)
            # delta_dict = mac.pu.collect_prob_deltas(ap_dict['label_probs_dict'],  **train_seq.formatted_data[split])
            # metric_data_dict['probability_deltas'] = delta_dict
            _metric_data_dict['label_probability_dict_train'] = ap_dict_train['label_probs_dict']
            _metric_data_dict['label_probability_dict_test'] = ap_dict_test['label_probs_dict']
            # visualize_average_probability.plot_probability_delta_matrix(delta_dict)
            # visualize_average_probability.plot_label_prob_dict(ap_dict['label_probs_dict'], mode='per_timestep')
            # plt.show()

        metric_summary_dict[count] = _metric_summary_dict
        metric_data_dict[count] = _metric_data_dict
        count += 1

path = f'../experiment_analysis_output/{tag_name}/'
os.makedirs(path, exist_ok=True)
if 'precision_recall' in metrics:
    if 'test_lower_threshold' in metrics['precision_recall']:
        with open(path + f'metric_summary_dict_pr_lower_threshold={metrics["precision_recall"]["test_lower_threshold"]}.pkl', 'wb') as f:
            pkl.dump(metric_summary_dict, f)
    else:
        with open(path + f'metric_summary_dict_pr.pkl', 'wb') as f:
            pkl.dump(metric_summary_dict, f)

if 'average_prob' in metrics:
    with open(path + 'metric_data_dict_average_prob.pkl', 'wb') as f:
        pkl.dump(metric_data_dict, f)

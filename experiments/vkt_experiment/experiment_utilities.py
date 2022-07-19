import config
import sys
import os
import torch
import pickle as pkl
import utilities as ut

script_path = sys.argv[0]
experiment_group_name = 'vkt_experiment'


def split_script_path(path):
    experiment_group_name = path.split('experiments/')[-1].split('/')[0]
    return experiment_group_name


def load_learner_data(learner_data_filename, get_training_data=False, device=None, normalize=False):
    path = os.path.join(config.ROOT, 'experiments', experiment_group_name, 'experiment_learner_data',
                        learner_data_filename + '/')
    learner_data = torch.load(path + 'learner_data.pth')
    with open(path + 'experiment_vars.pkl', 'rb') as f:
        experiment_vars = pkl.load(f)

    if get_training_data:
        training_data = ut.datasets.preprocessing.prep_data(experiment_vars['input_params']['dataset_name'],
                                                            experiment_vars['input_params']['dataset_params'], device,
                                                            normalize=False)

        return learner_data, experiment_vars, training_data

    return learner_data, experiment_vars


def index_dict_by_batch_indices(data_dict, batch_indices, device=None):

    data_dict_batch = {}
    for key in data_dict:
        if device is not None:
            data_dict_batch[key] = data_dict[key][batch_indices].to(device)
        else:
            data_dict_batch[key] = data_dict[key][batch_indices]

    return data_dict_batch


def get_key_from_learner_data(learner_data, mask_dict, target):
    return learner_data[:, :, mask_dict[target]]


def format_input_data(learner_data, mask_dict, sequence_length, data, labels, **kwargs):

    data_length = learner_data.shape[1]
    if sequence_length < data_length:
        # Data is stored such that x0, y0, r0, h0 are in v0, this means that vT+1 has padding for xT+1, yT+1, rT+1 but does
        # have a value for hT+1
        input_data = data[learner_data[:, :-1, mask_dict['supervised_ind']].type(torch.long)]
        supervision = labels[learner_data[:, :-1, mask_dict['supervised_ind']].type(torch.long)]
    else:
        input_data = data[learner_data[:, :, mask_dict['supervised_ind']].type(torch.long)]
        supervision = labels[learner_data[:, :, mask_dict['supervised_ind']].type(torch.long)]

    return dict(input_data=input_data, supervision=supervision)


# LOAD MODEL
def load_model(_experiment_group_name, experiment_name, experiment_save_specifier, filename):
    folder = f'{_experiment_group_name}/{experiment_name}/{experiment_save_specifier}/'
    model_dict = ut.file_utilities.load(folder, filename, return_dict=True)
    return model_dict

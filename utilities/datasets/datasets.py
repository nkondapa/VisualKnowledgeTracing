import torch
import config
import os
import numpy as np
import copy
import pickle as pkl
import utilities.datasets.preprocessing as preprocessing


def load_dataset(dataset_name, dataset_params):
    print(dataset_name)
    if dataset_name == 'OCT' or dataset_name == 'butterflies' or dataset_name == 'greebles':
        out = dataset_params['load_function'](**dataset_params)
    else:
        # if using gaussian blobs to test new methods
        data, labels = load_synthetic_data(dataset_name)
        out = dict(data=data, labels=labels)

    return out


def load_synthetic_data(dataset_name):
    path = os.path.join(config.SYNTHETIC_DATASET_FOLDER, dataset_name, 'data/')
    data = torch.FloatTensor(np.load(path + 'points.npy'))
    labels = torch.LongTensor(np.load(path + 'labels.npy'))

    return data, labels


def remap_labels(data, labels, remap_dict=None, target_labels=None, remove_unspecified=True):
    """

    :param data: the dataset, the first dimension must be indices.
        For example, if it is an image dataset it should be N,C,d1,d2.
    :param labels: the labels
    :param remap_dict: a dict that replaces the existing label with a new label
    :param target_labels: a list of labels that we want to keep and remap. Note if remap_dict is specified,
        this will lose priority.
    :param remove_unspecified: If True, remove labels that were not given a remapping. Otherwise group all remaining
        labels into their own class.
    :return: the dataset, the new labels, the old labels, the reversed dictionary (useful for plotting)
    """
    assert remap_dict is not None or target_labels is not None, 'One of remap_dict and target_labels must not be None.'

    # create remap dict from target labels or check that passed remap dict is indeed a dict
    if remap_dict is None:
        assert type(target_labels) is list, f'target labels must be a list - is {type(target_labels)}'
        remap_dict = {}
        for c, tg in enumerate(target_labels):
            remap_dict[int(tg)] = c
    else:
        assert type(remap_dict) is dict, f'remap dict must be dict - is {type(remap_dict)}'

    # setup for numpy or torch
    if type(labels) == torch.Tensor:
        tmp = torch.zeros(size=labels.shape)
    else:
        tmp = np.zeros(shape=labels.shape)

    # relabel points filling unspecified with a catch-all class
    class_id_for_else = len(remap_dict)
    key_list = list(remap_dict.keys())
    for key in key_list:
        if type(key) == str:
            remap_dict[int(key)] = remap_dict[key]
            del remap_dict[key]

    for i, label in enumerate(labels):
        if label.item() in remap_dict:
            tmp[i] = remap_dict[label.item()]
        else:
            tmp[i] = class_id_for_else

    #############################################################

    orig_labels = copy.copy(labels)
    if remove_unspecified:
        mask = tmp != len(remap_dict)
        labels = tmp[mask]
        orig_labels = orig_labels[mask]
        data = data[mask]

    label_names_dict = dict(zip(remap_dict.values(), remap_dict.keys()))
    return data, labels, orig_labels, label_names_dict

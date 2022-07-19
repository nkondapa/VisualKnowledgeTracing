import config
import os
import json
import pickle as pkl
import torch
import numpy as np
import torch


def save_model(data, folder_name, specifier):
    os.makedirs(os.path.join(config.TRAINED_MODELS_FOLDER, folder_name), exist_ok=True)
    path = os.path.join(config.TRAINED_MODELS_FOLDER, folder_name, specifier)

    if specifier.split('.')[-1] == 'pt':

        with open(path, 'wb') as f:
            torch.save(data, f)

    elif specifier.split('.')[-1] == 'pkl':
        with open(path, 'wb') as f:
            pkl.dump(data, f)

    else:
        raise ValueError('File format not recognized!')


def save_dict(d, folder_name, specifier, as_json=True):

    if as_json:
        if specifier.split('.')[-1] != 'json':
            specifier += '.json'
    else:
        if specifier.split('.')[-1] != 'pkl':
            specifier += '.pkl'

    os.makedirs(os.path.join(config.TRAINED_MODELS_FOLDER, folder_name), exist_ok=True)
    path = os.path.join(config.TRAINED_MODELS_FOLDER, folder_name, specifier)

    if as_json:
        with open(path, 'wb') as f:
            json.dump(d, f)
    else:
        with open(path, 'wb') as f:
            pkl.dump(d, f)


def load(folder_name, specifier, **kwargs):

    _accepted_formats = ['pt', 'json', 'pkl', 'npy']
    assert '.' in specifier and specifier.split('.')[-1] in _accepted_formats, \
        f'Format {specifier} not recognized. Accepted formats are: {_accepted_formats}'

    ext = specifier.split('.')[-1]

    path = os.path.join(config.TRAINED_MODELS_FOLDER, folder_name, specifier)

    if ext == 'json':
        with open(path, 'rb') as f:
            obj = json.load(f)
        return obj
    elif ext == 'pkl':
        with open(path, 'rb') as f:
            obj = pkl.load(f)
        return obj
    elif ext == 'pt':
        with open(path, 'rb') as f:
            obj = torch.load(f)
            if 'return_dict' in kwargs:
                return obj
        # This won't be needed once all model save reflect the desired dictionary structure {epoch, model}
        if type(obj) is dict:
            return obj['model']
        else:
            return obj
    elif ext == 'npy':
        with open(path, 'rb') as f:
            obj = np.load(f)
        return obj
    else:
        raise Exception()


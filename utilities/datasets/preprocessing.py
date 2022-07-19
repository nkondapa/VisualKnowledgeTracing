import torch
from config import device as default_device
from utilities import datasets as utd


def to_tensor(points, labels, device=default_device):

    points = torch.Tensor(points).to(device)
    labels = torch.LongTensor(labels).to(device)

    return points, labels


def add_bias_to_tensor(points):

    bias = torch.ones(size=(points.shape[0], 1)).to(points.device.type)
    points = torch.cat([points, bias], dim=1)

    return points


def one_hot(vec, num_classes, replace_zeros_with_neg_one=False):

    one_hot_labels = torch.nn.functional.one_hot(vec, num_classes)
    if replace_zeros_with_neg_one:
        one_hot_labels[one_hot_labels == 0] = -1

    return one_hot_labels


def prep_data_for_experiment(points, labels, num_classes, add_bias=True,
                             device=default_device, replace_zeros_with_neg_one=True):

    points, labels = to_tensor(points, labels, device)

    if add_bias:
        points = add_bias_to_tensor(points)

    one_hot_labels = one_hot(labels, num_classes, replace_zeros_with_neg_one)

    return points, labels, one_hot_labels


def prep_data(dataset_name, dataset_params, device=None, normalize=False):

    if 'normalize' not in dataset_params:
        dataset_params['normalize'] = normalize

    out = utd.datasets.load_dataset(dataset_name, dataset_params)
    data, labels = out['data'], out['labels']

    if dataset_params.get('add_bias', False):
        data = utd.preprocessing.add_bias_to_tensor(data)

    if device is not None:
        data, labels = to_tensor(data, labels, device)

    if 'remap_params' in dataset_params:
        data, labels, orig_labels, reverse_label_map = \
            utd.datasets.remap_labels(data, labels, **dataset_params['remap_params'])

        out = {'data': data, 'labels': labels, 'orig_labels': orig_labels, 'reverse_label_map': reverse_label_map}
        return out

    return out


def compute_bounds(points, symmetric=False):

    xbounds = [min(points[:, 0]), max(points[:, 0])]
    ybounds = [min(points[:, 1]), max(points[:, 1])]

    if symmetric:
        v = max(abs(xbounds[0]), abs(xbounds[1]))
        xbounds = [-v, v]
        v = max(abs(ybounds[0]), abs(ybounds[1]))
        ybounds = [-v, v]

    return xbounds, ybounds



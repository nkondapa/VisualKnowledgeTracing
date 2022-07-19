import utilities as ut
import numpy as np
import torch
import torchvision
import config


make_str_from_dict = lambda x: f"{x['feature_extractor_folder']}_{x['feature_extractor_filename']}"


def load_feature_extractor(feature_extractor_folder, feature_extractor_filename, device='cpu', eval=True):
    tmp = ut.file_utilities.load(feature_extractor_folder, feature_extractor_filename)
    if hasattr(tmp, 'embeddingnet'):
        net = tmp
        fe = tmp.embeddingnet
    elif hasattr(tmp, 'feature_extractor'):
        net = tmp
        fe = tmp.feature_extractor
    else:
        raise Exception('Unrecognized object structure!')
    if device is not None:
        net = net.to(device)
        fe = fe.to(device)
    if eval:
        net = net.eval().requires_grad_(False)
        fe = fe.eval().requires_grad_(False)

    return net, fe


def resnet_forward(resnet, data):

    print('computing resnet_forward...')
    resnet.to(config.device)
    _data_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224))])
    def data_transform(_data):
        _data = _data.repeat(1, 3, 1, 1)
        _data = _data_transform(_data).to(config.device)
        return _data

    num_ims = data.shape[0]
    batch_size = 16
    batch_inds = np.arange(0, num_ims, batch_size)
    batch_inds = np.concatenate([batch_inds, [num_ims]])
    feats = []
    for i in range(len(batch_inds) - 1):
        print(f'{batch_inds[i + 1]} / {num_ims}')
        ims = data[batch_inds[i]:batch_inds[i + 1]]
        x = resnet(data_transform(ims))
        feats.append(x.cpu())

    feats = torch.cat(feats, 0)
    return feats


def subsample_data(dataset_name, subsample_fraction=None, num_samples=5000, inds=None, dataset_params=None):

    if dataset_params is None:
        dataset_params = {'add_bias': False}
    data_dictionary = ut.datasets.preprocessing.prep_data(dataset_name, dataset_params)
    data = data_dictionary['data']
    labels = data_dictionary['labels']

    if num_samples is not None or subsample_fraction is not None:
        if inds is not None:
            data = data[inds]
            labels = labels[inds]
        elif num_samples is not None:
            if num_samples < data.shape[0]:
                inds = np.arange(data.shape[0])
                inds = np.random.choice(inds, num_samples, replace=False)
                data = data[inds]
                labels = labels[inds]
        elif subsample_fraction is not None:
            if subsample_fraction < 1.0:
                inds = np.arange(data.shape[0])
                num_samples = int(subsample_fraction * data.shape[0])
                inds = np.random.choice(inds, num_samples, replace=False)
                data = data[inds]
                labels = labels[inds]
        else:
            raise Exception('Parameters specified incorrectly!')

    return data, labels


def feature_extractors_expected_distance_on_dataset(fe1_params, fe2_params, dataset_name, subsample_fraction=None, num_samples=5000, inds=None):

    data, labels = subsample_data(dataset_name, subsample_fraction=None, num_samples=5000, inds=None)

    net1, fe1 = load_feature_extractor(**fe1_params)
    if 'resnet' in fe1_params['feature_extractor_folder']:
        x1 = resnet_forward(fe1, data)
    else:
        x1 = fe1(data)
    if type(fe2_params) is not list:
        fe2_params = [fe2_params]

    fe1_vector = torch.nn.utils.parameters_to_vector(fe1.parameters())
    distances = {}
    for fe2_params_i in fe2_params:
        net_i, fe2_i = load_feature_extractor(**fe2_params_i)
        if 'resnet' in fe2_params_i['feature_extractor_folder']:
            x2 = resnet_forward(fe2_i, data)
        else:
            x2 = fe2_i(data)
        fe2_i_vector = torch.nn.utils.parameters_to_vector(fe2_i.parameters())
        parameters_euclidean_distance = torch.linalg.vector_norm(fe1_vector - fe2_i_vector).mean().item()
        euclidean_distance = torch.linalg.vector_norm(x1 - x2, dim=1).mean().item()
        cosine_distance = (1 - torch.nn.functional.cosine_similarity(x1, x2, dim=1)).mean().item()
        distances_i = dict(euclidean_distance=euclidean_distance, cosine_distance=cosine_distance,
                 parameters_euclidean_distance=parameters_euclidean_distance)
        if hasattr(net_i, 'classify'):
            acc = (net_i.classify(x2).argmax(1) == labels).type(torch.float).mean().item()
            distances_i['feature_extractor2_acc'] = acc
        distances[f"{make_str_from_dict(fe1_params)} - {make_str_from_dict(fe2_params_i)}"] = distances_i

    return distances
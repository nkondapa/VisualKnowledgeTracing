from tracing_models.lstm import *
from tracing_models.baseline import *
from tracing_models.dkt_translation import *
import torch
import numpy as np
import experiments.vkt_experiment.analysis_utilities.probability_utilities as pu
import copy
import time


def execute_class_conditioned_probability(model, formatted_data, training_data, model_out, **kwargs):

    func = None
    if isinstance(model, GTBaseline):
        func = _gt_baseline_model
    elif isinstance(model, Baseline):
        func = _baseline_model
    elif isinstance(model, TimestepBaseline):
        func = _timestep_baseline_model
    elif isinstance(model, LSTMObserverModelDR1v3):
        func = _lstm_observer_model_dr_1v3
    elif isinstance(model, LSTMObserverModelCLF1v2):
        func = _lstm_observer_model_clf_1v2
    elif isinstance(model, LSTMObserverModelCLF1v2wPCA):
        func = _lstm_observer_model_clf_1v2_wpca
    else:
        raise NotImplementedError(f'{type(model)} not yet supported!')

    rng = np.random.default_rng()
    subsample_indices = None
    ## COMPUTATIONAL LOAD IS HIGH FOR THESE METHODS, SUBSAMPLE TRAINING DATA TO REDUCE TIME COST
    if 'subsample_fraction' in kwargs or 'subsample_indices' in kwargs:
        if 'subsample_indices' in kwargs:
            subsample_indices = kwargs['subsample_indices']
        else:
            subsample_fraction = kwargs['subsample_fraction']
            indices = np.arange(training_data['data'].shape[0])
            subsample_indices = rng.choice(indices, size=int(subsample_fraction * len(indices)), replace=False)

        training_data = training_data.copy()
        training_data['data'] = training_data['data'][subsample_indices]
        training_data['labels'] = training_data['labels'][subsample_indices]

    return func(model, **formatted_data, training_data=training_data, **model_out)


def _gt_baseline_model(model, training_data, **kwargs):

    bsz, seq_len = kwargs['data_t'].shape[:2]
    labels = training_data['labels']
    responses = torch.nn.functional.one_hot(labels)[(None,)*2].repeat(bsz, seq_len, 1, 1).type(torch.float)
    probs = torch.softmax(responses, dim=-1)
    label_probs_dict = pu.generate_label_probs_dict(labels, probs)

    return dict(responses=responses, probs=probs, label_probs_dict=label_probs_dict)


def _baseline_model(model, training_data, **kwargs):

    bsz, seq_len = kwargs['data_t'].shape[:2]
    data = training_data['data']
    labels = training_data['labels']
    feats = model.feature_extraction(data)
    responses = model.classifier(feats)
    probs = torch.softmax(responses, dim=-1)[(None,)*2].repeat(bsz, seq_len, 1, 1)

    label_probs_dict = pu.generate_label_probs_dict(labels, probs)

    return dict(responses=responses, probs=probs, label_probs_dict=label_probs_dict)


def _timestep_baseline_model(model, training_data, **kwargs):

    bsz, seq_len = kwargs['data_t'].shape[:2]
    data = training_data['data']
    labels = training_data['labels']
    feats = model.feature_extraction(data)
    responses = feats @ model.hyperplanes[-1].T
    probs = torch.softmax(responses, dim=-1).unsqueeze(0).unsqueeze(0).repeat(bsz, seq_len, 1, 1)
    label_probs_dict = pu.generate_label_probs_dict(labels, probs)

    return dict(responses=responses, probs=probs, label_probs_dict=label_probs_dict)


def _lstm_observer_model_dr_1v3(model, last_response, last_hidden_state, teaching_signal_t,
                                      last_cell_state, training_data, **kwargs):

    # This model requires the query's class label and the image embedding
    device = last_response.device
    bsz, _, _, num_classes = last_response.shape
    data = training_data['data'].to(device)
    labels = training_data['labels'].to(device)
    teaching_signal = torch.nn.functional.one_hot(labels, num_classes).unsqueeze(0).repeat(bsz, 1, 1)
    orig_shape = data.shape
    num_queries = orig_shape[0]
    last_response = last_response.repeat(1, num_queries, 1, 1).squeeze()
    feats = model.feature_extraction(data).reshape(1, num_queries, -1).repeat(bsz, 1, 1)
    responses = []
    for k in range(bsz):
        query_responses = []
        for qi in range(num_queries):
            x = torch.cat([feats[k, None, qi], teaching_signal[k, None, qi],
                       last_response[k, None, qi]], dim=-1).unsqueeze(1)
            out, _ = model.network(x, (last_hidden_state[:, [k]], last_cell_state[:, [k]]))
            response = model.transform_hidden_state_to_response(out)
            query_responses.append(response)

        responses.append(torch.stack(query_responses, dim=1))
    responses = torch.stack(responses, dim=0).squeeze(3).repeat(1, teaching_signal_t.shape[1], 1, 1)
    probs = torch.softmax(responses, dim=-1)

    label_probs_dict = pu.generate_label_probs_dict(labels, probs)

    return dict(responses=responses, label_probs_dict=label_probs_dict, probs=probs)


def _lstm_observer_model_clf_1v2(model, feats, responses_t, hidden_states, cell_states, training_data, **kwargs):

    # LSTMObserverModelDR1v2
    device = responses_t.device
    data = training_data['data'].to(device)
    labels = training_data['labels'].to(device)
    unique_labels = torch.unique(labels)
    num_classes = len(unique_labels)
    orig_shape = responses_t.shape
    bsz = orig_shape[0]
    seq_len = orig_shape[1]
    response_pad = torch.zeros_like(responses_t[:, 0]).repeat(1, num_classes, 1)
    teaching_signal = torch.nn.functional.one_hot(unique_labels).unsqueeze(0).repeat(bsz, 1, 1)
    feats_pad = torch.zeros_like(feats[:, 0]).repeat(1, num_classes, 1)
    test_feats = model.feature_extraction(data).unsqueeze(0).repeat(bsz, 1, 1)

    st = time.time()
    x = torch.cat([feats[:, -1].repeat(1, num_classes, 1), teaching_signal,
                   responses_t[:, -1].repeat(1, num_classes, 1)], dim=-1).flatten(0, 1).unsqueeze(1)

    h_state = hidden_states[-1, :].repeat(1, num_classes, 1)
    c_state = cell_states[-1, :].repeat(1, num_classes, 1)
    out, _ = model.network(x, (h_state, c_state))
    tmp = model.transform_hidden_state_to_hyperplane(out).reshape(bsz, num_classes, *model.out_dimension_w_bias)
    hyperplane, bias = tmp[:, :, :, :-1], tmp[:, :, :, -1]
    response = torch.zeros(bsz, test_feats.shape[1], num_classes)
    for li, label in enumerate(unique_labels):
        mask = label == labels
        _response = torch.einsum('bcd, bqd -> bqc', hyperplane[:, li], test_feats[:, mask]) + bias[:, [li]].repeat(1, mask.sum(), 1)
        response[:, mask] = _response

    responses = response.reshape(bsz, 1, test_feats.shape[1], num_classes).repeat(1, seq_len, 1, 1)
    probs = torch.softmax(responses, dim=-1)

    label_probs_dict = pu.generate_label_probs_dict(labels, probs)
    print(time.time() - st)
    return dict(responses=responses, label_probs_dict=label_probs_dict, probs=probs)



def _lstm_observer_model_clf_1v2_wpca(model, feats, responses_t, hidden_states, cell_states, training_data,
                                      per_class_accuracies, **kwargs):

    # LSTMObserverModelDR1v2
    device = responses_t.device
    data = training_data['data'].to(device)
    labels = training_data['labels'].to(device)
    unique_labels = torch.unique(labels)
    num_classes = len(unique_labels)
    orig_shape = responses_t.shape
    bsz = orig_shape[0]
    seq_len = orig_shape[1]
    response_pad = torch.zeros_like(responses_t[:, 0]).repeat(1, num_classes, 1)
    teaching_signal = torch.nn.functional.one_hot(unique_labels).unsqueeze(0).repeat(bsz, 1, 1)
    feats_pad = torch.zeros_like(feats[:, 0]).repeat(1, num_classes, 1)
    test_feats = model.feature_extraction(data).unsqueeze(0).repeat(bsz, 1, 1)
    pca = per_class_accuracies.repeat(1, 1, num_classes, 1)

    st = time.time()
    x = torch.cat([feats[:, -1].repeat(1, num_classes, 1), pca[:, -1], teaching_signal,
                   responses_t[:, -1].repeat(1, num_classes, 1)], dim=-1).flatten(0, 1).unsqueeze(1)

    h_state = hidden_states[-1, :].repeat(1, num_classes, 1)
    c_state = cell_states[-1, :].repeat(1, num_classes, 1)
    out, _ = model.network(x, (h_state, c_state))
    tmp = model.transform_hidden_state_to_hyperplane(out).reshape(bsz, num_classes, *model.out_dimension_w_bias)
    hyperplane, bias = tmp[:, :, :, :-1], tmp[:, :, :, -1]
    response = torch.zeros(bsz, test_feats.shape[1], num_classes)
    for li, label in enumerate(unique_labels):
        mask = label == labels
        _response = torch.einsum('bcd, bqd -> bqc', hyperplane[:, li], test_feats[:, mask]) + bias[:, [li]].repeat(1, mask.sum(), 1)
        response[:, mask] = _response

    responses = response.reshape(bsz, 1, test_feats.shape[1], num_classes).repeat(1, seq_len, 1, 1)
    probs = torch.softmax(responses, dim=-1)

    label_probs_dict = pu.generate_label_probs_dict(labels, probs)
    print(time.time() - st)
    return dict(responses=responses, label_probs_dict=label_probs_dict, probs=probs)
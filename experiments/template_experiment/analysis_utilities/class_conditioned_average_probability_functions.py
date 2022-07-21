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
    elif isinstance(model, LSTMObserverModelDR1v2):
        func = _lstm_observer_model_dr_1v2
    elif isinstance(model, LSTMObserverModelDR1v3):
        func = _lstm_observer_model_dr_1v3
    elif isinstance(model, LSTMObserverModelCLF1v2):
        func = _lstm_observer_model_clf_1v2
    elif isinstance(model, LSTMObserverModelCLF1v2wPCA):
        func = _lstm_observer_model_clf_1v2_wpca
    else:
        raise NotImplementedError(f'{type(model)} not yet supported!')

    subsample_indices = None
    ## COMPUTATIONAL LOAD IS HIGH FOR THESE METHODS, SUBSAMPLE TRAINING DATA TO REDUCE TIME COST
    if 'subsample_fraction' in kwargs or 'subsample_indices' in kwargs:
        if 'subsample_indices' in kwargs:
            subsample_indices = kwargs['subsample_indices']
        else:
            subsample_fraction = kwargs['subsample_fraction']
            indices = np.arange(training_data['data'].shape[0])
            subsample_indices = np.random.choice(indices, size=int(subsample_fraction * len(indices)), replace=False)

        training_data = training_data.copy()
        training_data['data'] = training_data['data'][subsample_indices]
        training_data['labels'] = training_data['labels'][subsample_indices]

    return func(model, **formatted_data, training_data=training_data, **model_out), subsample_indices


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
    responses = []
    for t in range(model.hyperplanes.shape[0]):
        response = feats @ model.hyperplanes[t].T
        responses.append(response)

    responses = torch.stack(responses, dim=0)
    probs = torch.softmax(responses, dim=-1).unsqueeze(0).repeat(bsz, 1, 1, 1)
    label_probs_dict = pu.generate_label_probs_dict(labels, probs)

    return dict(responses=responses, probs=probs, label_probs_dict=label_probs_dict)


def _dkt_plus_translation(model, last_qa, state_out, qid, **kwargs):
    num_classes = kwargs['probs'].shape[-1]
    future_y_mat = last_qa.unsqueeze(-2).repeat(1, 1, 5, 1)
    for i in range(num_classes):
        fill_vec = torch.zeros(num_classes)
        fill_vec[i] = 1
        future_y_mat[:, :, i, -num_classes:] = fill_vec

    probs = []
    for i in range(num_classes):
        out, _ = model.rnn(future_y_mat[:, :, i], state_out)
        logits = model.fc(out)
        prob = torch.softmax(logits, dim=-1)
        probs.append(prob)

    prob_matrix = torch.stack(probs, dim=-2).repeat(1, qid.shape[1], 1, 1)
    probs = torch.gather(prob_matrix, dim=2, index=qid.argmax(2).unsqueeze(-1).repeat(1, 1, 1, num_classes))
    return dict(probs=probs)


def _lstm_observer_model_dr_1v2(model, feats, responses_t, hidden_states, cell_states, training_data, **kwargs):

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
    responses = []
    for t in range(seq_len):
        print(t)
        if t == 0:
            x = torch.cat([feats_pad, teaching_signal,
                           response_pad], dim=-1).flatten(0, 1).unsqueeze(1)
        else:
            x = torch.cat([feats[:, t - 1].repeat(1, num_classes, 1), teaching_signal,
                           responses_t[:, t-1].repeat(1, num_classes, 1)], dim=-1).flatten(0, 1).unsqueeze(1)

        h_state = hidden_states[t, :].repeat(1, num_classes, 1)
        c_state = cell_states[t, :].repeat(1, num_classes, 1)
        out, _ = model.network(x, (h_state, c_state))
        c_response = model.transform_hidden_state_to_response(out).reshape(bsz, num_classes, 1, num_classes)
        response = c_response[:, labels]
        # bqc response.reshape(bsz, c, 1, c)
        responses.append(response)

    responses = torch.stack(responses, dim=1).reshape(bsz, seq_len, test_feats.shape[1], num_classes)
    probs = torch.softmax(responses, dim=-1)

    label_probs_dict = pu.generate_label_probs_dict(labels, probs)
    print(time.time() - st)
    return dict(responses=responses, label_probs_dict=label_probs_dict, probs=probs)


def _lstm_observer_model_dr_1v3(model, feats, responses_t, hidden_states, cell_states, training_data, **kwargs):

    # LSTMObserverModelDR1v3
    device = responses_t.device
    data = training_data['data'].to(device)
    labels = training_data['labels'].to(device)
    num_queries = data.shape[0]
    orig_shape = responses_t.shape
    bsz = orig_shape[0]
    seq_len = orig_shape[1]
    response_pad = torch.zeros_like(responses_t[:, 0]).repeat(1, num_queries, 1)

    feats = model.feature_extraction(data).unsqueeze(0).repeat(bsz, 1, 1)
    teaching_signal = torch.nn.functional.one_hot(labels, responses_t.shape[-1]).unsqueeze(0).repeat(bsz, 1, 1)

    responses = []
    for t in range(seq_len):
        print(t)
        if t == 0:
            x = torch.cat([feats, teaching_signal,
                           response_pad], dim=-1).flatten(0, 1).unsqueeze(1)
        else:
            x = torch.cat([feats, teaching_signal,
                           responses_t[:, t-1].repeat(1, num_queries, 1)], dim=-1).flatten(0, 1).unsqueeze(1)

        h_state = hidden_states[t, :].repeat(1, num_queries, 1)
        c_state = cell_states[t, :].repeat(1, num_queries, 1)
        out, _ = model.network(x, (h_state, c_state))
        response = model.transform_hidden_state_to_response(out)
        responses.append(response.reshape(bsz, num_queries, -1))

    responses = torch.stack(responses, dim=1)
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
    responses = []
    for t in range(seq_len):
        print(t)
        if t == 0:
            x = torch.cat([feats_pad, teaching_signal,
                           response_pad], dim=-1).flatten(0, 1).unsqueeze(1)
        else:
            x = torch.cat([feats[:, t - 1].repeat(1, num_classes, 1), teaching_signal,
                           responses_t[:, t-1].repeat(1, num_classes, 1)], dim=-1).flatten(0, 1).unsqueeze(1)

        h_state = hidden_states[t, :].repeat(1, num_classes, 1)
        c_state = cell_states[t, :].repeat(1, num_classes, 1)
        out, _ = model.network(x, (h_state, c_state))
        tmp = model.transform_hidden_state_to_hyperplane(out).reshape(bsz, num_classes, *model.out_dimension_w_bias)
        hyperplane, bias = tmp[:, :, :, :-1], tmp[:, :, :, -1]
        response = torch.zeros(bsz, test_feats.shape[1], num_classes)
        for li, label in enumerate(unique_labels):
            mask = label == labels
            _response = torch.einsum('bcd, bqd -> bqc', hyperplane[:, li], test_feats[:, mask]) + bias[:, [li]].repeat(1, mask.sum(), 1)
            response[:, mask] = _response
        responses.append(response)

    responses = torch.stack(responses, dim=1).reshape(bsz, seq_len, test_feats.shape[1], num_classes)
    probs = torch.softmax(responses, dim=-1)

    label_probs_dict = pu.generate_label_probs_dict(labels, probs)
    print(time.time() - st)
    return dict(responses=responses, label_probs_dict=label_probs_dict, probs=probs)


def _lstm_observer_model_clf_1v2_wpca(model, feats, responses_t, hidden_states, cell_states, training_data, per_class_accuracies, **kwargs):

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
    responses = []
    for t in range(seq_len):
        print(t)
        if t == 0:
            x = torch.cat([feats_pad, pca[:, t], teaching_signal,
                           response_pad], dim=-1).flatten(0, 1).unsqueeze(1)
        else:
            x = torch.cat([feats[:, t - 1].repeat(1, num_classes, 1), pca[:, t], teaching_signal,
                           responses_t[:, t-1].repeat(1, num_classes, 1)], dim=-1).flatten(0, 1).unsqueeze(1)

        h_state = hidden_states[t, :].repeat(1, num_classes, 1)
        c_state = cell_states[t, :].repeat(1, num_classes, 1)
        out, _ = model.network(x, (h_state, c_state))
        tmp = model.transform_hidden_state_to_hyperplane(out).reshape(bsz, num_classes, *model.out_dimension_w_bias)
        hyperplane, bias = tmp[:, :, :, :-1], tmp[:, :, :, -1]
        response = torch.zeros(bsz, test_feats.shape[1], num_classes)
        for li, label in enumerate(unique_labels):
            mask = label == labels
            _response = torch.einsum('bcd, bqd -> bqc', hyperplane[:, li], test_feats[:, mask]) + bias[:, [li]].repeat(1, mask.sum(), 1)
            response[:, mask] = _response
        responses.append(response)

    responses = torch.stack(responses, dim=1).reshape(bsz, seq_len, test_feats.shape[1], num_classes)
    probs = torch.softmax(responses, dim=-1)

    label_probs_dict = pu.generate_label_probs_dict(labels, probs)
    print(time.time() - st)
    return dict(responses=responses, label_probs_dict=label_probs_dict, probs=probs, per_class_accuracies=per_class_accuracies)


def _lstm_observer_model1(model, responses_t, hidden_states, cell_states, training_data, **kwargs):

    # LSTMObserverModel1
    device = responses_t.device
    data = training_data['data'].to(device)
    labels = training_data['labels'].to(device)
    num_queries = data.shape[0]
    orig_shape = responses_t.shape
    bsz = orig_shape[0]
    seq_len = orig_shape[1]
    response_pad = torch.zeros_like(responses_t[:, 0]).repeat(1, num_queries, 1)

    feats = model.feature_extraction(data).unsqueeze(0).repeat(bsz, 1, 1)
    teaching_signal = torch.nn.functional.one_hot(labels, responses_t.shape[-1]).unsqueeze(0).repeat(bsz, 1, 1)

    responses = []
    for t in range(seq_len):
        print(t)
        if t == 0:
            x = torch.cat([feats, teaching_signal,
                           response_pad], dim=-1).flatten(0, 1).unsqueeze(1)
        else:
            x = torch.cat([feats, teaching_signal,
                           responses_t[:, t-1].repeat(1, num_queries, 1)], dim=-1).flatten(0, 1).unsqueeze(1)

        h_state = hidden_states[t, :].repeat(1, num_queries, 1)
        c_state = cell_states[t, :].repeat(1, num_queries, 1)
        out, _ = model.network(x, (h_state, c_state))
        responses.append(out[:, :, model.hidden_state_dimension:].reshape(bsz, num_queries, -1))

    responses = torch.stack(responses, dim=1)
    probs = torch.softmax(responses, dim=-1)

    label_probs_dict = pu.generate_label_probs_dict(labels, probs)

    return dict(responses=responses, label_probs_dict=label_probs_dict, probs=probs)

def _lstm_observer_model1v1(model, responses_t, hidden_states, cell_states, training_data, **kwargs):

    # LSTMObserverModel1
    device = responses_t.device
    data = training_data['data'].to(device)
    labels = training_data['labels'].to(device)
    num_queries = data.shape[0]
    orig_shape = responses_t.shape
    bsz = orig_shape[0]
    seq_len = orig_shape[1]
    response_pad = torch.zeros_like(responses_t[:, 0]).repeat(1, num_queries, 1)

    feats = model.feature_extraction(data).unsqueeze(0).repeat(bsz, 1, 1)
    teaching_signal = torch.nn.functional.one_hot(labels, responses_t.shape[-1]).unsqueeze(0).repeat(bsz, 1, 1)

    responses = []
    for t in range(seq_len):
        print(t)
        if t == 0:
            x = torch.cat([feats, teaching_signal,
                           response_pad], dim=-1).flatten(0, 1).unsqueeze(1)
        else:
            x = torch.cat([feats, teaching_signal,
                           responses_t[:, t-1].repeat(1, num_queries, 1)], dim=-1).flatten(0, 1).unsqueeze(1)

        h_state = hidden_states[t, :].repeat(1, num_queries, 1)
        c_state = cell_states[t, :].repeat(1, num_queries, 1)
        out, _ = model.network(x, (h_state, c_state))
        response = model.transform_hidden_state_to_response(out)
        responses.append(response.reshape(bsz, num_queries, -1))

    responses = torch.stack(responses, dim=1)
    probs = torch.softmax(responses, dim=-1)

    label_probs_dict = pu.generate_label_probs_dict(labels, probs)

    return dict(responses=responses, label_probs_dict=label_probs_dict, probs=probs)

def _lstm_observer_model1v2(model, data_t, last_teaching_signal, last_response, last_hidden_state,
                                      last_cell_state, **kwargs):

    raise NotImplementedError
    # LSTMObserverModel1v2
    orig_shape = data_t.shape
    feats = model.feature_extraction(data_t.flatten(0, -4)).reshape(*orig_shape[:2], -1)
    responses = []
    for t in range(feats.shape[1]):
        x = torch.cat([feats[:, t].flatten(start_dim=1), last_teaching_signal.flatten(start_dim=1),
                       last_response.flatten(start_dim=1)], dim=-1).unsqueeze(1)
        out, _ = model.network(x, (last_hidden_state, last_cell_state))
        responses.append(out[:, :, model.hidden_state_dimension:])

    responses = torch.stack(responses, dim=1)
    return dict(responses=responses)


def _lstm_observer_model2(model, responses_t, hidden_states, cell_states, training_data, **kwargs):

    # LSTMObserverModel2
    device = responses_t.device
    data = training_data['data'].to(device)
    labels = training_data['labels'].to(device)
    num_queries = data.shape[0]
    orig_shape = responses_t.shape
    bsz = orig_shape[0]
    seq_len = orig_shape[1]
    response_pad = torch.zeros_like(responses_t[:, 0]).repeat(1, num_queries, 1)

    feats = model.feature_extraction(data).unsqueeze(0).repeat(bsz, 1, 1)
    teaching_signal = torch.nn.functional.one_hot(labels, responses_t.shape[-1]).unsqueeze(0).repeat(bsz, 1, 1)

    responses = []
    for t in range(seq_len):
        print(t)
        if t == 0:
            x = torch.cat([feats, teaching_signal,
                           response_pad], dim=-1).flatten(0, 1).unsqueeze(1)
        else:
            x = torch.cat([feats, teaching_signal,
                           responses_t[:, t-1].repeat(1, num_queries, 1)], dim=-1).flatten(0, 1).unsqueeze(1)

        h_state = hidden_states[t, :].repeat(1, num_queries, 1)
        c_state = cell_states[t, :].repeat(1, num_queries, 1)
        out, _ = model.network(x, (h_state, c_state))
        cut = out.reshape(bsz, num_queries, 1, -1)[:, :, 0, model.hidden_state_dimension:]
        hyperplanes = cut[:, :, :np.prod(model.out_dimension)].reshape(bsz, num_queries, *model.out_dimension)
        trans_feats = cut[:, :, np.prod(model.out_dimension):]
        response = torch.einsum('bqcd, bqd -> bqc', hyperplanes, trans_feats)
        responses.append(response)

    responses = torch.stack(responses, dim=1)
    probs = torch.softmax(responses, dim=-1)

    label_probs_dict = pu.generate_label_probs_dict(labels, probs)

    return dict(responses=responses, label_probs_dict=label_probs_dict, probs=probs)


def _lstm_observer_model3(model, responses_t, hidden_states, cell_states, training_data, **kwargs):

    # LSTMObserverModel3
    device = responses_t.device
    data = training_data['data'].to(device)
    labels = training_data['labels'].to(device)
    num_queries = data.shape[0]
    orig_shape = responses_t.shape
    bsz = orig_shape[0]
    seq_len = orig_shape[1]
    response_pad = torch.zeros_like(responses_t[:, 0]).repeat(1, num_queries, 1)

    feats = model.feature_extraction(data).unsqueeze(0).repeat(bsz, 1, 1)
    teaching_signal = torch.nn.functional.one_hot(labels, responses_t.shape[-1]).unsqueeze(0).repeat(bsz, 1, 1)

    st = time.time()
    responses = []
    for t in range(seq_len):
        print(t)
        if t == 0:
            x = torch.cat([feats, teaching_signal,
                           response_pad], dim=-1).flatten(0, 1).unsqueeze(1)
        else:
            x = torch.cat([feats, teaching_signal,
                           responses_t[:, t-1].repeat(1, num_queries, 1)], dim=-1).flatten(0, 1).unsqueeze(1)

        h_state = hidden_states[t, :].repeat(1, num_queries, 1)
        c_state = cell_states[t, :].repeat(1, num_queries, 1)
        out, _ = model.network(x, (h_state, c_state))
        hyperplanes = out[:, :, model.hidden_state_dimension:].reshape(bsz, num_queries, *model.out_dimension)
        response = torch.einsum('bqcd, bqd -> bqc', hyperplanes, feats)
        responses.append(response)

    responses = torch.stack(responses, dim=1)
    probs = torch.softmax(responses, dim=-1)

    label_probs_dict = pu.generate_label_probs_dict(labels, probs)
    print(time.time() - st)
    return dict(responses=responses, label_probs_dict=label_probs_dict, probs=probs)


def _lstm_observer_model3v2(model, responses_t, hidden_states, cell_states, teaching_signal_t, training_data, **kwargs):

    # LSTMObserverModel3v2
    device = responses_t.device
    data = training_data['data'].to(device)
    labels = training_data['labels'].to(device)
    num_queries = data.shape[0]
    orig_shape = responses_t.shape
    bsz = orig_shape[0]
    seq_len = orig_shape[1]
    teaching_signal_t = teaching_signal_t.unsqueeze(2)
    response_pad = torch.zeros_like(responses_t[:, 0]).repeat(1, num_queries, 1)
    teaching_pad = torch.zeros_like(teaching_signal_t[:, 0]).repeat(1, num_queries, 1)

    feats = model.feature_extraction(data).unsqueeze(0).repeat(bsz, 1, 1)
    # teaching_signal = torch.nn.functional.one_hot(labels, responses_t.shape[-1]).unsqueeze(0).repeat(bsz, 1, 1)

    st = time.time()
    responses = []
    for t in range(seq_len):
        print(t)
        if t == 0:
            x = torch.cat([feats, teaching_pad,
                           response_pad], dim=-1).flatten(0, 1).unsqueeze(1)
        else:
            x = torch.cat([feats, teaching_signal_t[:, t - 1].repeat(1, num_queries, 1),
                           responses_t[:, t - 1].repeat(1, num_queries, 1)], dim=-1).flatten(0, 1).unsqueeze(1)

        h_state = hidden_states[t, :].repeat(1, num_queries, 1)
        c_state = cell_states[t, :].repeat(1, num_queries, 1)
        out, _ = model.network(x, (h_state, c_state))
        hyperplanes = out[:, :, model.hidden_state_dimension:].reshape(bsz, num_queries, *model.out_dimension)
        response = torch.einsum('bqcd, bqd -> bqc', hyperplanes, feats)
        responses.append(response)

    responses = torch.stack(responses, dim=1)
    probs = torch.softmax(responses, dim=-1)

    label_probs_dict = pu.generate_label_probs_dict(labels, probs)
    print(time.time() - st)
    return dict(responses=responses, label_probs_dict=label_probs_dict, probs=probs)


def _lstm_observer_model3v3(model, responses_t, feats, hidden_states, cell_states, training_data, **kwargs):

    # LSTMObserverModel3v3
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
    responses = []
    for t in range(seq_len):
        print(t)
        if t == 0:
            x = torch.cat([feats_pad, teaching_signal,
                           response_pad], dim=-1).flatten(0, 1).unsqueeze(1)
        else:
            x = torch.cat([feats[:, t - 1].repeat(1, num_classes, 1), teaching_signal,
                           responses_t[:, t-1].repeat(1, num_classes, 1)], dim=-1).flatten(0, 1).unsqueeze(1)

        h_state = hidden_states[t, :].repeat(1, num_classes, 1)
        c_state = cell_states[t, :].repeat(1, num_classes, 1)
        out, _ = model.network(x, (h_state, c_state))
        hyperplanes = out[:, :, model.hidden_state_dimension:].reshape(bsz, num_classes, *model.out_dimension)
        response = torch.zeros(bsz, test_feats.shape[1], num_classes)
        for li, label in enumerate(unique_labels):
            mask = label == labels
            _response = torch.einsum('bcd, bqd -> bqc', hyperplanes[:, li], test_feats[:, mask])
            response[:, mask] = _response
        responses.append(response)

    responses = torch.stack(responses, dim=1)
    probs = torch.softmax(responses, dim=-1)

    label_probs_dict = pu.generate_label_probs_dict(labels, probs)
    print(time.time() - st)
    return dict(responses=responses, label_probs_dict=label_probs_dict, probs=probs)
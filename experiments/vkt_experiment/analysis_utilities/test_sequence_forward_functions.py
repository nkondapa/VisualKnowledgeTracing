from observer_models.lstm import *
from observer_models.baseline import *
from observer_models.transformer import DRTransformerSAINT
from observer_models.dkt_translation import DKTTranslation
import torch
import numpy as np


def execute_test_seq_forward(model, formatted_data, model_out):

    func = None
    if isinstance(model, GTBaseline):
        func = _gt_baseline_model
    elif isinstance(model, Baseline):
        func = _baseline_model
    elif isinstance(model, TimestepBaseline):
        func = _timestep_baseline_model
    elif isinstance(model, DKTTranslation):
        func = _dkt_plus_translation
    elif isinstance(model, LSTMObserverModelDR1v1):
        func = _lstm_observer_model_dr_1v1
    elif isinstance(model, LSTMObserverModelDR1v2):
        func = _lstm_observer_model_dr_1v2
    elif isinstance(model, LSTMObserverModelDR1v3):
        func = _lstm_observer_model_dr_1v3
    elif isinstance(model, LSTMObserverModelCLF1v1):
        func = _lstm_observer_model_clf_1v1
    elif isinstance(model, LSTMObserverModelCLF1v2):
        func = _lstm_observer_model_clf_1v2
    elif isinstance(model, LSTMObserverModelCLF1v3):
        func = _lstm_observer_model_clf_1v3
    elif isinstance(model, DRTransformerSAINT):
        func = _transformer_observer_model_dr
    elif isinstance(model, ExemplarBaseline):
        func = _exemplar_baseline_model
    elif isinstance(model, PrototypeBaseline):
        func = _prototype_baseline_model
    elif isinstance(model, PerClassAccuracyBaseline):
        func = _per_class_accuracy_baseline_model
    elif isinstance(model, LSTMObserverModelCLF1v2wPCA):
        func = _lstm_observer_model_clf_1v2_wpca
    elif isinstance(model, LSTMObserverModelDR1v3wPCA):
        func = _lstm_observer_model_dr_1v3_wpca
    else:
        raise NotImplementedError(f'{type(model)} not yet supported!')

    return func(model, **formatted_data, **model_out)


def _gt_baseline_model(model, data_t, teaching_signal_t, **kwargs):

    responses = teaching_signal_t.clone().unsqueeze(-2).type(torch.float)
    return dict(responses=responses)


def _baseline_model(model, data_t, **kwargs):

    responses = []
    for t in range(data_t.shape[1]):
        feat = model.feature_extraction(data_t[:, t, 0])
        response = model.classifier(feat)
        responses.append(response)

    responses = torch.stack(responses, dim=1)
    return dict(responses=responses)


def _per_class_accuracy_baseline_model(model, data_t, per_class_accuracies, **kwargs):

    responses = []
    for t in range(data_t.shape[1]):
        feat = model.feature_extraction(data_t[:, t, 0])
        pca_feats = torch.cat([feat, per_class_accuracies[:, -1, 0]], dim=-1)
        _out = model.conditional_transform(pca_feats)
        response = model.classifier(_out)
        responses.append(response)

    responses = torch.stack(responses, dim=1)
    return dict(responses=responses)


def _timestep_baseline_model(model, data_t, **kwargs):

    responses = []
    for t in range(data_t.shape[1]):
        feat = model.feature_extraction(data_t[:, t, 0])
        response = feat @ model.hyperplanes[-1].T
        responses.append(response)

    responses = torch.stack(responses, dim=1)
    return dict(responses=responses, hyperplanes=model.hyperplanes[-1].T.repeat(1, 1, data_t.shape[1]))


def _prototype_baseline_model(model, data_t, prototypes, prototype_counts, **kwargs):
    orig_shape = data_t.shape
    feats = model.feature_extraction(data_t.flatten(0, -4)).reshape(*orig_shape[:2], -1)
    average_prototypes = prototypes / prototype_counts.clone()

    dist = torch.sqrt(torch.sum((feats.unsqueeze(2).repeat(1, 1, prototypes.shape[-2], 1) - average_prototypes.unsqueeze(1)) ** 2, dim=-1)).unsqueeze(-2)
    score = torch.exp(model.c * dist * -1).detach()
    responses = score
    return dict(responses=responses)


def _exemplar_baseline_model(model, data_t, sims, teaching_signal, feats, **kwargs):

    # teaching_signal comes from train data out dict
    orig_shape = data_t.shape
    test_feats = model.feature_extraction(data_t.flatten(0, -4)).reshape(*orig_shape[:2], -1)
    all_feats = torch.cat([feats, test_feats.unsqueeze(-2)], dim=1)
    # distances = torch.stack([model.pairwise_euclidean_distance(all_feats[i, :, 0]) for i in range(all_feats.shape[0])])
    distances = torch.stack(
        [torch.cdist(all_feats[i, :, 0], all_feats[i, :, 0], compute_mode='donot_use_mm_for_euclid_dist') for i in
         range(feats.shape[0])])

    # distances[distances <= 0] = 1e-8

    distances = distances.sqrt()
    sims = torch.exp(-model.c * distances)
    training_len = feats.shape[1]
    responses = []
    for test_ind in range(orig_shape[1]):
        test_ind += training_len
        numers = torch.zeros(size=(data_t.shape[0], teaching_signal.shape[-1])).to(feats.device) + 1e-8
        for c in range(teaching_signal.shape[-1]):
            sel_sims = sims[:, :training_len, test_ind] * teaching_signal[:, :training_len, c]
            numers[:, c] = numers[:, c] + sel_sims.sum(-1)
        tmp = numers ** model.gamma
        score = (tmp) / (tmp.sum(-1).unsqueeze(-1))
        responses.append(score)

    responses = torch.stack(responses, dim=1).unsqueeze(-2).detach()
    return dict(responses=responses)


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


def _transformer_observer_model_dr(model, data_t, teaching_signal_t, responses_t, encoder_first, decoder_first, **kwargs):
    orig_shape = data_t.shape
    feats = model.feature_extraction(data_t.flatten(0, -4)).reshape(*orig_shape[:2], 1, -1)
    test_encoder_out = torch.cat([feats.squeeze(-2), teaching_signal_t], dim=-1)

    responses = torch.zeros(size=responses_t.shape)
    for t in range(data_t.shape[1]):
        ## pass through each of the encoder blocks in sequence
        first_block = True
        encoder_out = torch.cat([encoder_first[:, :-1], test_encoder_out[:, [t]]], dim=1)
        for x in range(model.num_encoder_layers):
            if x >= 1:
                first_block = False
            encoder_out = model.encoder[x](encoder_out, first_block=first_block)

        ## pass through each decoder blocks in sequence
        first_block = True
        # decoder_out = decoder_first
        decoder_out = decoder_first[:, :-1]
        for x in range(model.num_decoder_layers):
            if x >= 1:
                first_block = False
            decoder_out = model.decoder[x](decoder_out, encoder_out=encoder_out, first_block=first_block)

        ## Output layer
        _resp = model.out(decoder_out).unsqueeze(-2)
        responses[:, t] = _resp[:, -1]

    # [25, 15, 1, 5]
    return dict(responses=responses)


def _lstm_observer_model_dr_1v1(model, data_t, responses, **kwargs):

    # this model doesn't any use any information about the currently presented query
    responses = responses[:,  [-1]].repeat(1, data_t.shape[1], 1, 1)
    return dict(responses=responses)


def _lstm_observer_model_dr_1v2(model, data_t, teaching_signal_t, last_feat, last_response, last_hidden_state,
                                      last_cell_state, **kwargs):

    # This model requires the query's class label
    orig_shape = data_t.shape
    responses = []
    for t in range(teaching_signal_t.shape[1]):
        x = torch.cat([last_feat.flatten(start_dim=1), teaching_signal_t[:, t].flatten(start_dim=1),
                       last_response.flatten(start_dim=1)], dim=-1).unsqueeze(1)
        out, _ = model.network(x, (last_hidden_state, last_cell_state))
        response = model.transform_hidden_state_to_response(out)
        responses.append(response)

    responses = torch.stack(responses, dim=1)
    return dict(responses=responses)


def _lstm_observer_model_dr_1v3(model, data_t, teaching_signal_t, last_feat, last_response, last_hidden_state,
                                      last_cell_state, **kwargs):

    # This model requires the query's class label and the image embedding
    orig_shape = data_t.shape
    feats = model.feature_extraction(data_t.flatten(0, -4)).reshape(*orig_shape[:2], -1)
    responses = []
    for t in range(feats.shape[1]):
        x = torch.cat([feats[:, t].flatten(start_dim=1), teaching_signal_t[:, t].flatten(start_dim=1),
                       last_response.flatten(start_dim=1)], dim=-1).unsqueeze(1)
        out, _ = model.network(x, (last_hidden_state, last_cell_state))
        response = model.transform_hidden_state_to_response(out)
        responses.append(response)

    responses = torch.stack(responses, dim=1)
    return dict(responses=responses)


def _lstm_observer_model_dr_1v3_wpca(model, data_t, teaching_signal_t, last_feat, last_response, last_hidden_state,
                                      last_cell_state, per_class_accuracies, **kwargs):

    # This model requires the query's class label and the image embedding
    orig_shape = data_t.shape
    feats = model.feature_extraction(data_t.flatten(0, -4)).reshape(*orig_shape[:2], -1)
    responses = []
    for t in range(feats.shape[1]):
        x = torch.cat([feats[:, t].flatten(start_dim=1), per_class_accuracies[:, -1].flatten(start_dim=1),
                       teaching_signal_t[:, t].flatten(start_dim=1),
                       last_response.flatten(start_dim=1)], dim=-1).unsqueeze(1)
        out, _ = model.network(x, (last_hidden_state, last_cell_state))
        response = model.transform_hidden_state_to_response(out)
        responses.append(response)

    responses = torch.stack(responses, dim=1)
    return dict(responses=responses)


def _lstm_observer_model_clf_1v1(model, data_t, hyperplanes, bias, **kwargs):
    # this model doesn't any use any information about the currently presented query to
    # predict the hyperplane, but the query is multiplied to it to produce the response
    orig_shape = data_t.shape
    feats = model.feature_extraction(data_t.flatten(0, -4)).reshape(*orig_shape[:2], 1, -1)
    responses = torch.einsum('bscd, bsqd -> bsqc', hyperplanes[:, [-1]], feats) + bias[:, [-1]]
    return dict(responses=responses)


def _lstm_observer_model_clf_1v2(model, data_t, teaching_signal_t, last_feat, last_response, last_hidden_state,
                                last_cell_state, **kwargs):

    # This model requires the query's class label to predict a hyperplane
    orig_shape = data_t.shape
    feats = model.feature_extraction(data_t.flatten(0, -4)).reshape(*orig_shape[:2], 1, -1)
    hyperplanes_bias = []
    for t in range(teaching_signal_t.shape[1]):
        x = torch.cat([last_feat.flatten(start_dim=1), teaching_signal_t[:, t].flatten(start_dim=1),
                       last_response.flatten(start_dim=1)], dim=-1).unsqueeze(1)
        out, _ = model.network(x, (last_hidden_state, last_cell_state))
        tmp = model.transform_hidden_state_to_hyperplane(out).reshape(orig_shape[0], *model.out_dimension_w_bias)
        hyperplanes_bias.append(tmp)

    hyperplanes_bias = torch.stack(hyperplanes_bias, dim=1)
    hyperplanes = hyperplanes_bias[:, :, :, :-1]
    bias = hyperplanes_bias[:, :, :, -1].unsqueeze(-2)
    responses = torch.einsum('bscd, bsqd -> bsqc', hyperplanes, feats) + bias

    return dict(responses=responses)


def _lstm_observer_model_clf_1v2_wpca(model, data_t, teaching_signal_t, last_feat, last_response, last_hidden_state,
                                last_cell_state, per_class_accuracies, **kwargs):

    # This model requires the query's class label to predict a hyperplane
    orig_shape = data_t.shape
    feats = model.feature_extraction(data_t.flatten(0, -4)).reshape(*orig_shape[:2], 1, -1)
    hyperplanes_bias = []
    for t in range(teaching_signal_t.shape[1]):
        x = torch.cat([last_feat.flatten(start_dim=1), per_class_accuracies[:, -1].flatten(start_dim=1),
                       teaching_signal_t[:, t].flatten(start_dim=1),
                       last_response.flatten(start_dim=1)], dim=-1).unsqueeze(1)
        out, _ = model.network(x, (last_hidden_state, last_cell_state))
        tmp = model.transform_hidden_state_to_hyperplane(out).reshape(orig_shape[0], *model.out_dimension_w_bias)
        hyperplanes_bias.append(tmp)

    hyperplanes_bias = torch.stack(hyperplanes_bias, dim=1)
    hyperplanes = hyperplanes_bias[:, :, :, :-1]
    bias = hyperplanes_bias[:, :, :, -1].unsqueeze(-2)
    responses = torch.einsum('bscd, bsqd -> bsqc', hyperplanes, feats) + bias

    return dict(responses=responses)


def _lstm_observer_model_clf_1v3(model, data_t, teaching_signal_t, last_feat, last_response, last_hidden_state,
                                last_cell_state, **kwargs):

    # This model requires the query's class label and the query's embedding to predict a hyperplane
    orig_shape = data_t.shape
    feats = model.feature_extraction(data_t.flatten(0, -4)).reshape(*orig_shape[:2], 1, -1)
    hyperplanes_bias = []
    for t in range(teaching_signal_t.shape[1]):
        x = torch.cat([feats[:, t].flatten(start_dim=1),
                       teaching_signal_t[:, t].flatten(start_dim=1),
                       last_response.flatten(start_dim=1)], dim=-1).unsqueeze(1)
        out, _ = model.network(x, (last_hidden_state, last_cell_state))
        tmp = model.transform_hidden_state_to_hyperplane(out).reshape(orig_shape[0], *model.out_dimension_w_bias)
        hyperplanes_bias.append(tmp)

    hyperplanes_bias = torch.stack(hyperplanes_bias, dim=1)
    hyperplanes = hyperplanes_bias[:, :, :, :-1]
    bias = hyperplanes_bias[:, :, :, -1].unsqueeze(-2)
    responses = torch.einsum('bscd, bsqd -> bsqc', hyperplanes, feats) + bias

    return dict(responses=responses)
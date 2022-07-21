from torch import nn
import torch
import numpy as np
import utilities as ut
from tracing_models.lstm import fe_builder


class LSTMTracingModel(nn.Module):

    def __init__(self, response_dimension, input_dimension, out_dimension, seq_len, pretrained_feature_extractor=None,
                 hidden_state_dimension=256, num_layers=3, init_loss_weight=0, dropout=0.0, information_level='score',
                 **kwargs):
        super(LSTMTracingModel, self).__init__()

        self.response_dimension = response_dimension
        self.input_dimension = input_dimension
        self.out_dimension = out_dimension
        self.out_dimension_w_bias = out_dimension[0], out_dimension[1] + 1
        self.seq_len = seq_len
        self.information_level = information_level
        self.hidden_state_dimension = hidden_state_dimension
        self.init_loss_weight = init_loss_weight
        self.feature_extractor = fe_builder.get_feature_extractor(pretrained_feature_extractor, self.input_dimension)
        assert self.feature_extractor.dimension == self.input_dimension[1]

        self.feature_extractor_trainable_state = True

        self.network = nn.LSTM(input_size=np.prod(self.response_dimension) * 2 + np.prod(self.input_dimension),
                               hidden_size=self.hidden_state_dimension,
                               batch_first=True,
                               num_layers=num_layers, dropout=dropout)

        self.transform_hidden_state_to_hyperplane = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.hidden_state_dimension, np.prod(self.out_dimension)),
            nn.PReLU(),
            nn.Linear(np.prod(self.out_dimension), np.prod(self.out_dimension_w_bias))
        )

        self.network_trainable_state = True
        self.loss = ut.information_level_functions.get_model_loss_function_for_information_level(self.information_level,
                                                                                                 **{
                                                                                                     'num_classes':
                                                                                                         self.out_dimension[
                                                                                                             0],
                                                                                                     **kwargs})

    def set_network_trainable(self, state=True):
        self.network = self.network.requires_grad_(state)
        self.network_trainable_state = state

    def set_feature_extractor_trainable(self, state=True):
        self.feature_extractor = self.feature_extractor.requires_grad_(state)
        self.feature_extractor_trainable_state = state

    def low_gpu_memory_feature_extraction(self, data):
        data_vec = []
        batch_inds = np.arange(0, data.shape[0], 128)
        for i in range(len(batch_inds)):
            print(i)
            if i == len(batch_inds) - 1:
                batch = data[batch_inds[i]:]
            else:
                batch = data[batch_inds[i]:batch_inds[i + 1]]
            batch = batch.to(self.feature_extractor.fc_embed.weight.device)
            data_vec.append(self.feature_extraction(batch))
            torch.cuda.empty_cache()
        data = torch.cat(data_vec)
        return data

    def feature_extraction(self, data):
        return self.feature_extractor(data)

    def forward(self, responses_t, data_t, teaching_signal_t, **kwargs):

        device = responses_t.device

        bsz = responses_t.shape[0]
        hidden_states = []
        cell_states = []
        hyperplanes_bias = []

        out_c = None
        hidden_state = None
        orig_shape = data_t.shape
        feats = self.feature_extraction(data_t.flatten(0, -4)).reshape(*orig_shape[:2], 1, -1)
        response_pad = torch.zeros_like(responses_t[:, 0])

        for t in range(self.seq_len):

            if t == 0:
                x = torch.cat([feats[:, t].flatten(start_dim=1), teaching_signal_t[:, t].flatten(start_dim=1),
                               response_pad.flatten(start_dim=1)], dim=-1).unsqueeze(1)
                out, (hidden_state, out_c) = self.network(x)
            else:
                x = torch.cat([feats[:, t].flatten(start_dim=1), teaching_signal_t[:, t].flatten(start_dim=1),
                               responses_t[:, t - 1].flatten(start_dim=1)], dim=-1).unsqueeze(1)
                out, (hidden_state, out_c) = self.network(x, (hidden_state, out_c))

            tmp = self.transform_hidden_state_to_hyperplane(out).reshape(bsz, *self.out_dimension_w_bias)
            hyperplanes_bias.append(tmp)

            hidden_states.append(hidden_state)
            cell_states.append(out_c)

        hyperplanes_bias = torch.stack(hyperplanes_bias, dim=1)
        hyperplanes = hyperplanes_bias[:, :, :, :-1]
        bias = hyperplanes_bias[:, :, :, -1].unsqueeze(-2)
        init_hyperplanes = hyperplanes[:, 0]

        hidden_states = torch.stack(hidden_states, dim=0)
        cell_states = torch.stack(cell_states, dim=0)

        # This model will estimate one future state, so we drop that for the responses in the loss
        responses = torch.einsum('bscd, bsqd -> bsqc', hyperplanes, feats) + bias
        init_responses = responses[:, 0]

        return dict(hyperplanes=hyperplanes, bias=bias, init_hyperplanes=init_hyperplanes, cell_states=cell_states,
                    hidden_states=hidden_states, last_cell_state=out_c, last_hidden_state=hidden_state,
                    init_responses=init_responses, feats=feats.detach(),
                    last_feat=feats[:, [-1]],
                    last_response=responses_t[:, [-1]], last_teaching_signal=teaching_signal_t[:, [-1]],
                    responses=responses)

    def loss_function(self, out, responses_t, **kwargs):

        responses = out['responses']
        init_responses = out['init_responses']
        target = responses_t
        crit_loss = self.loss(responses, target)
        init_loss = self.loss(init_responses, target[:, 0])
        loss = crit_loss.mean() + self.init_loss_weight * init_loss.mean()
        loss_vals = dict(crit_loss_mean=crit_loss.mean().item(), init_loss=init_loss.mean().item())
        return loss, loss_vals

    def format_data(self, input_data, supervision, learner_data, mask_dict, split_dict, device=None):
        if device is None:
            device = self.network.all_weights[0][0].device.type
        input_data = input_data.to(device)
        supervision = supervision.to(device)
        learner_data = learner_data.to(device)
        num_classes = self.response_dimension[-1]
        responses = ut.information_level_functions.convert_responses_to_information_level(learner_data, mask_dict,
                                                                                          num_classes,
                                                                                          level=self.information_level)

        supervision = ut.datasets.preprocessing.one_hot(supervision.squeeze(), num_classes=num_classes)
        data_t = input_data
        responses_t = responses[:, :self.seq_len]
        teaching_signal_t = supervision

        formatted_data = {}
        for split in split_dict:
            spind = split_dict[split]
            formatted_data[split] = dict(data_t=data_t[spind],  # class_responses_t=class_responses_t[spind],
                                         responses_t=responses_t[spind], teaching_signal_t=teaching_signal_t[spind])

        return formatted_data

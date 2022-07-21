from torch import nn
import torch
import numpy as np
import utilities as ut
from tracing_models.template import fe_builder


class Template(nn.Module):

    # add whatever params are needed for the model
    def __init__(self, response_dimension, input_dimension, out_dimension, seq_len, pretrained_feature_extractor=None,
                 information_level='argmax',
                 **kwargs):
        super(Template, self).__init__()

        self.response_dimension = response_dimension
        self.input_dimension = input_dimension
        self.out_dimension = out_dimension
        self.out_dimension_w_bias = out_dimension[0], out_dimension[1] + 1
        self.seq_len = seq_len
        self.information_level = information_level

        self.feature_extractor = fe_builder.get_feature_extractor(pretrained_feature_extractor, self.input_dimension)
        assert self.feature_extractor.dimension == self.input_dimension[1]

        self.feature_extractor_trainable_state = True

        self.network = None

        self.network_trainable_state = True

        # get a loss function that matches the response type (ie. fitting rankings will require a different loss)
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

    def feature_extraction(self, data):
        return self.feature_extractor(data)

    def forward(self, responses_t, data_t, teaching_signal_t, **kwargs):

        device = responses_t.device
        orig_shape = data_t.shape
        feats = self.feature_extraction(data_t.flatten(0, -4)).reshape(*orig_shape[:2], 1, -1)

        for t in range(self.seq_len):
            break

        responses = None

        # this dict goes to the loss function (out[])
        return dict(responses=responses)

    def loss_function(self, out, responses_t, **kwargs):

        # compute the loss
        responses = out['responses']
        target = responses_t
        crit_loss = self.loss(responses, target)
        loss = crit_loss.mean()  # total loss
        # dict to store multiple losses (if there are multiple)
        loss_vals = dict(crit_loss_mean=crit_loss.mean().item())

        return loss, loss_vals

    def format_data(self, input_data, supervision, learner_data, mask_dict, split_dict, device=None):
        if device is None:
            device = self.network.all_weights[0][0].device.type

        # move data to same device as network
        input_data = input_data.to(device)
        supervision = supervision.to(device)
        learner_data = learner_data.to(device)

        # convert response to information level (mostly for simulated experiments) but has something for
        # getting rankings rather than the argmax
        num_classes = self.response_dimension[-1]
        responses = ut.information_level_functions.convert_responses_to_information_level(learner_data, mask_dict,
                                                                                          num_classes,
                                                                                          level=self.information_level)

        # convert supervision to one hot labels
        supervision = ut.datasets.preprocessing.one_hot(supervision.squeeze(), num_classes=num_classes)

        data_t = input_data
        responses_t = responses[:, :self.seq_len]
        teaching_signal_t = supervision

        # split data according to the train val test splits
        formatted_data = {}
        for split in split_dict:
            spind = split_dict[split]
            # these are the keys for the data in the forward loop
            formatted_data[split] = dict(data_t=data_t[spind],
                                         responses_t=responses_t[spind], teaching_signal_t=teaching_signal_t[spind])

        return formatted_data

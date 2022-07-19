from torch import nn
import torch
import numpy as np
import utilities as ut
from observer_models.lstm import fe_builder


class Baseline(nn.Module):

    def __init__(self, response_dimension, input_dimension, out_dimension, seq_len, pretrained_feature_extractor=None,
                 information_level='score', **kwargs):
        super(Baseline, self).__init__()

        self.response_dimension = response_dimension
        self.input_dimension = input_dimension
        self.out_dimension = out_dimension
        self.seq_len = seq_len
        self.information_level = information_level
        self.feature_extractor = fe_builder.get_feature_extractor(pretrained_feature_extractor, self.input_dimension)
        assert self.feature_extractor.dimension == self.input_dimension[1]

        self.classifier = nn.Linear(self.out_dimension[1], self.out_dimension[0])

        self.feature_extractor_trainable_state = True

        self.loss = ut.information_level_functions.get_model_loss_function_for_information_level(self.information_level,
                                                                                                 **{
                                                                                                     'num_classes':
                                                                                                         self.out_dimension[
                                                                                                             0],
                                                                                                     **kwargs})

    def set_feature_extractor_trainable(self, state=True):
        self.feature_extractor = self.feature_extractor.requires_grad_(state)
        self.feature_extractor_trainable_state = state

    def reparametrize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

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

        bsz = responses_t.shape[0]
        device = responses_t.device

        feats = []
        for t in range(self.seq_len):
            feat = self.feature_extraction(data_t[:, t, 0])
            feats.append(feat)

        feats = torch.stack(feats, dim=1).unsqueeze(2)
        responses = self.classifier(feats)
        init_responses = responses[:, 0]
        return dict(init_responses=init_responses, feats=feats.detach(),
                    responses=responses)

    def loss_function(self, out, responses_t, **kwargs):

        responses = out['responses']
        init_responses = out['init_responses']
        target = responses_t
        crit_loss = self.loss(responses, target)
        init_loss = self.loss(init_responses, target[:, 0])
        loss = crit_loss.mean()
        loss_vals = dict(crit_loss_mean=crit_loss.mean().item(), init_loss=init_loss.mean().item())
        return loss, loss_vals

    def format_data(self, input_data, supervision, learner_data, mask_dict, split_dict, device=None):
        if device is None:
            device = self.classifier.weight.device.type
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

    def predict_test_sequence_performance(self, data_t, **kwargs):

        responses = []
        for t in range(data_t.shape[1]):
            feat = self.feature_extraction(data_t[:, t, 0])
            response = self.classifier(feat)
            responses.append(response)

        responses = torch.stack(responses, dim=1)
        return dict(response=responses)

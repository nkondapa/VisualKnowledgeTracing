from torch import nn
import torch
import numpy as np
import utilities as ut
from observer_models.lstm import fe_builder


class ExemplarBaseline(nn.Module):

    def __init__(self, response_dimension, input_dimension, out_dimension, seq_len, pretrained_feature_extractor=None,
                 information_level='score', **kwargs):
        super(ExemplarBaseline, self).__init__()

        self.response_dimension = response_dimension
        self.input_dimension = input_dimension
        self.out_dimension = out_dimension
        self.seq_len = seq_len
        self.information_level = information_level
        self.feature_extractor = fe_builder.get_feature_extractor(pretrained_feature_extractor, self.input_dimension)
        assert self.feature_extractor.dimension == self.input_dimension[1]

        self.feature_extractor_trainable_state = True
        self.c = torch.nn.Parameter(torch.FloatTensor([1]))
        self.gamma = torch.nn.Parameter(torch.FloatTensor([1]))
        self.loss = ut.information_level_functions.get_model_loss_function_for_information_level(self.information_level,
                                                                                                 **{'num_classes': self.out_dimension[0],
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

    def pairwise_euclidean_distance(self, x):
        y = x.clone()
        x_norm = x.norm(dim=1, keepdim=True)
        y_norm = y.norm(dim=1).T
        distance = x_norm * x_norm + y_norm * y_norm - 2 * x.mm(y.T)
        return distance

    def forward(self, responses_t, data_t, teaching_signal_t, **kwargs):

        bsz = responses_t.shape[0]
        device = responses_t.device
        orig_shape = data_t.shape
        feats = self.feature_extraction(data_t.flatten(0, -4)).reshape(*orig_shape[:2], 1, -1)
        distances = torch.stack(
            [torch.cdist(feats[i, :, 0], feats[i, :, 0], compute_mode='donot_use_mm_for_euclid_dist') for i in
             range(feats.shape[0])])

        distances = distances.sqrt()
        sims = torch.exp(-self.c * distances)

        responses = []
        for t in range(responses_t.shape[1]):
            numers = torch.zeros(size=(responses_t.shape[0], responses_t.shape[-1])).to(feats.device) + 1e-8
            if t != 0:
                for c in range(responses_t.shape[-1]):
                    sel_sims = sims[:, :t, t] * teaching_signal_t[:, :t, c]
                    numers[:, c] = numers[:, c] + sel_sims.sum(-1)
                tmp = numers ** self.gamma
                score = (tmp) / (tmp.sum(-1).unsqueeze(-1))
            else:
                score = numers
            responses.append(score)

        responses = torch.stack(responses, dim=1).unsqueeze(-2)
        init_responses = responses[:, [0]]
        return dict(init_responses=init_responses, feats=feats.detach(),
                    responses=responses, sims=sims, teaching_signal=teaching_signal_t)

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
            device = self.feature_extractor.feature_extractor[0].weight.device.type
        self.c = self.c.to(device)
        self.gamma = self.gamma.to(device)
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

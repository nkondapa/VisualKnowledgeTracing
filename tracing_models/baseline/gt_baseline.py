from torch import nn
import torch
import config
import utilities as ut


class GTBaseline(nn.Module):

    def __init__(self, response_dimension, input_dimension, out_dimension, seq_len, pretrained_feature_extractor=None,
                 information_level='score', **kwargs):
        super(GTBaseline, self).__init__()

        self.response_dimension = response_dimension
        self.input_dimension = input_dimension
        self.out_dimension = out_dimension
        self.seq_len = seq_len
        self.information_level = information_level
        self.loss = ut.information_level_functions.get_model_loss_function_for_information_level(self.information_level,
                                                                                                 **{'num_classes':
                                                                                                        self.out_dimension[
                                                                                                            0],
                                                                                                    **kwargs})

    def forward(self, responses_t, data_t, teaching_signal_t, **kwargs):
        responses = teaching_signal_t.clone().unsqueeze(-2).type(torch.float)
        return dict(responses=responses, init_responses=responses[:, 0])

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
        device = config.device
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
            formatted_data[split] = dict(data_t=data_t[spind],
                                         responses_t=responses_t[spind], teaching_signal_t=teaching_signal_t[spind])

        return formatted_data

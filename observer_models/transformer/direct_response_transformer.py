from torch import nn
import torch
import numpy as np
import copy
import utilities as ut
from observer_models.lstm import fe_builder
from observer_models.transformer.subclasses import EncoderBlock, DecoderBlock


class DRTransformerSAINT(nn.Module):

    def __init__(self, response_dimension, input_dimension, out_dimension, seq_len, pretrained_feature_extractor=None,
                 hidden_state_dimension=256, num_encoder_layers=2, num_decoder_layers=2, encoder_heads=8,
                 decoder_heads=8,
                 init_loss_weight=0, dropout=0.0, information_level='score',
                 **kwargs):
        super(DRTransformerSAINT, self).__init__()

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

        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.encoder_heads = encoder_heads
        self.decoder_heads = decoder_heads

        encoder_in_dim = self.feature_extractor.dimension + np.prod(self.response_dimension)
        decoder_in_dim = np.prod(self.response_dimension)

        self.encoder = self.get_clones(
            EncoderBlock(encoder_in_dim, hidden_state_dimension, self.encoder_heads, seq_len),
            self.num_encoder_layers)
        self.decoder = self.get_clones(
            DecoderBlock(decoder_in_dim, hidden_state_dimension, self.decoder_heads, seq_len),
            self.num_decoder_layers)

        self.out = nn.Linear(in_features=hidden_state_dimension, out_features=np.prod(self.response_dimension) + 0)

        self.network_trainable_state = True
        self.loss = ut.information_level_functions.get_model_loss_function_for_information_level(self.information_level,
                                                                                                 **{
                                                                                                     'num_classes':
                                                                                                         self.out_dimension[
                                                                                                             0],
                                                                                                     **kwargs})

    def get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

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
        ## pass through each of the encoder blocks in sequence
        first_block = True
        orig_shape = data_t.shape
        feats = self.feature_extraction(data_t.flatten(0, -4)).reshape(*orig_shape[:2], 1, -1)
        encoder_first = torch.cat([feats.squeeze(-2), teaching_signal_t], dim=-1)
        encoder_out = torch.cat([feats.squeeze(-2), teaching_signal_t], dim=-1)
        for x in range(self.num_encoder_layers):
            if x >= 1:
                first_block = False
            encoder_out = self.encoder[x](encoder_out, first_block=first_block)

        ## pass through each decoder blocks in sequence
        first_block = True
        decoder_out = responses_t.type(torch.float).squeeze(-2)
        # response information to the decoder is delayed a step
        decoder_first = torch.cat([torch.zeros_like(decoder_out[:, [0], :]), decoder_out], dim=1)
        decoder_out = torch.cat([torch.zeros_like(decoder_out[:, [0], :]), decoder_out], dim=1)[:, :-1]
        for x in range(self.num_decoder_layers):
            if x >= 1:
                first_block = False
            decoder_out = self.decoder[x](decoder_out, encoder_out=encoder_out, first_block=first_block)

        ## Output layer
        responses = self.out(decoder_out).unsqueeze(-2)
        init_responses = responses[:, 0]

        return dict(init_responses=init_responses, feats=feats.detach(),
                    encoder_first=encoder_first, decoder_first=decoder_first,
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
            device = self.out.weight.device.type
        input_data = input_data.to(device)
        supervision = supervision.to(device)
        learner_data = learner_data.to(device)
        num_classes = self.response_dimension[-1]
        # This needs to be fixed for human data, it doesn't really make sense if the data isn't the full information and
        # is being reduced
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

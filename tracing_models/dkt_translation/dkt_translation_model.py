import torch
from torch import nn
import utilities as ut


# -*- coding:utf-8 -*-
"""
    Paper reference: Addressing Two Problems in Deep Knowledge Tracing via
    Prediction-Consistent Regularization (https://arxiv.org/pdf/1806.02180.pdf)
"""


class DKTTranslation(nn.Module):
    def __init__(
        self,
        embed_dim,
        input_dim,
        hidden_dim,
        layer_num,
        output_dim,
        gamma, reg1, reg2,
        device="cpu",
        cell_type="lstm",
    ):
        """ The first deep knowledge tracing network architecture.

        :param embed_dim: int, the embedding dim for each skill.
        :param input_dim: int, the number of skill(question) * 2.
        :param hidden_dim: int, the number of hidden state dim.
        :param layer_num: int, the layer number of the sequence number.
        :param output_dim: int, the number of skill(question).
        :param device: str, 'cpu' or 'cuda:0', the default value is 'cpu'.
        :param cell_type: str, the sequence model type, it should be 'lstm', 'rnn' or 'gru'.
        """
        super(DKTTranslation, self).__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.output_dim = output_dim
        self.device = device
        self.cell_type = cell_type
        self.rnn = None
        self.loss_fn = DKTLoss(gamma, reg1, reg2)

        # self.fc = nn.Linear(self.hidden_dim, self.output_dim + 1)  # +1 is a holdover if padding is needed
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

        if cell_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                self.embed_dim, self.hidden_dim, self.layer_num, batch_first=True
            )
        elif cell_type.lower() == "rnn":
            self.rnn = nn.RNN(
                self.embed_dim, self.hidden_dim, self.layer_num, batch_first=True
            )
        elif cell_type.lower() == "gru":
            self.rnn = nn.GRU(
                self.embed_dim, self.hidden_dim, self.layer_num, batch_first=True
            )

        if self.rnn is None:
            raise ValueError("cell type only support lstm, rnn or gru type.")

    def forward(self, qa, qid, state_in=None, **kwargs):
        """

        :param x: The input is a tensor(int64) with 2 dimension, like [H, k]. H is the batch size,
        k is the length of user's skill/question id sequence.
        :param state_in: optional. The state tensor for sequence model.
        :return:
        """
        h0 = torch.zeros(
            (self.layer_num, qa.size(0), self.hidden_dim), device=self.device
        )
        c0 = torch.zeros(
            (self.layer_num, qa.size(0), self.hidden_dim), device=self.device
        )

        if state_in is None:
            state_in = (h0, c0)

        state, state_out = self.rnn(qa, state_in)
        logits = self.fc(state)
        probs = torch.softmax(logits, dim=-1)
        out = dict(logits=logits, state_out=state_out, probs=probs, last_qa=qa[:, [-1]])
        return out

    def format_data(self, input_data, supervision, learner_data, mask_dict, split_dict, device=None):
        if device is None:
            device = self.rnn.all_weights[0][0].device
            self.device = device

        input_data = input_data.to(device)
        supervision = supervision.to(device)
        teaching_signal_t = ut.datasets.preprocessing.one_hot(supervision.squeeze(), num_classes=self.output_dim)
        learner_data = learner_data.to(device)
        class_response = learner_data[:, :, mask_dict['class_response']].type(torch.int)
        if class_response.shape[1] > supervision.shape[1]:
            class_response = class_response[:, :-1]

        qid = torch.nn.functional.one_hot(supervision.squeeze(), self.output_dim)
        responses_t = torch.nn.functional.one_hot(class_response.type(torch.long), self.output_dim)

        qa = torch.cat([qid[:, :-1], responses_t.squeeze(-2)[:, :-1], qid[:, 1:]], dim=-1).type(torch.float)
        qa = torch.cat([torch.zeros_like(qa[:, [0]]) - 1, qa], dim=1)

        mask = torch.ones_like(qid)

        formatted_data = {}
        for split in split_dict:
            spind = split_dict[split]
            formatted_data[split] = dict(qa=qa[spind], qid=qid.view(*qid.shape, 1)[spind],
                                         responses_t=responses_t[spind], teaching_signal_t=teaching_signal_t[spind],
                                         mask=mask[spind])

        return formatted_data

    def loss_function(self, out, responses_t, qid, mask, device='cuda', **kwargs):
        probs = out['probs']
        loss = self.loss_fn(probs, responses_t, qid, mask, device=device)
        return loss


class DKTLoss(nn.Module):
    def __init__(self, gamma=0.1, reg1=0.03, reg2=1.0, reduce=None):
        super(DKTLoss, self).__init__()
        self.reduce = reduce
        self.loss1 = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, probs, responses_t, qid, mask, device="cpu"):

        total = torch.sum(mask) + 1
        loss1 = self.loss1(probs.flatten(0, -2), responses_t.flatten(0, -2).argmax(-1)).sum()

        if self.reduce is None or self.reduce == "mean":
            loss1 = loss1 / total

        if self.reduce is not None and self.reduce not in ["mean", "sum"]:
            raise ValueError("the reduce should be mean or sum")

        return loss1
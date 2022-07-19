import torch


class HingeLoss:

    def __init__(self, margin=1.0, reduction='none'):
        self.margin = margin
        self.reduction = reduction

    def __call__(self, pred, target):
        if pred.flatten().shape != target.flatten().shape:
            raise ValueError('Prediction and target should have the same shape!')

        v = self.margin - target * pred
        v[v < 0] = 0

        if self.reduction == 'mean':
            v = v.mean()
        elif self.reduction == 'sum':
            v = v.sum()
        elif self.reduction == 'none':
            pass
        else:
            raise NotImplementedError(f'Reduction : {self.reduction} not implemented.')

        return v


class CrossEntropyWrapper:

    def __init__(self, reduction='none', **kwargs):
        self.loss = torch.nn.CrossEntropyLoss(reduction='none', **kwargs)

    def __call__(self, responses, target):
        v = self.loss(responses.flatten(0, -2), torch.argmax(target.flatten(0, -2), dim=-1))
        return v.reshape(*responses.shape[:-1])


class LabelSmoothing(torch.nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.default_confidence = 1.0 - smoothing
        self.default_smoothing = smoothing

    def forward(self, responses, target, smoothing=None):
        if smoothing is None:
            smoothing = self.default_smoothing
            confidence = self.default_confidence
        else:
            confidence = 1.0 - smoothing
        response_shape = responses.shape
        responses = responses.flatten(0, -2)
        target = torch.argmax(target.flatten(0, -2), dim=-1)
        logprobs = torch.nn.functional.log_softmax(responses, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        v = confidence * nll_loss + smoothing * smooth_loss
        loss = v.reshape(*response_shape[:-1])
        return loss


def cross_entropy_wrapper(responses, target):
    _loss = torch.nn.CrossEntropyLoss(reduction='none')
    v = _loss(responses.flatten(0, -2), torch.argmax(target.flatten(0, -2), dim=-1))
    return v.reshape(*responses.shape[:-1])


def softmax_mse_wrapper(responses, target):
    responses = torch.nn.functional.softmax(responses, dim=-1)
    _loss = torch.nn.MSELoss(reduction='none')
    return _loss(responses, target)
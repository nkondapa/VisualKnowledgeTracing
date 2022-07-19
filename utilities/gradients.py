import torch
import utilities as ut


def compute_loss_gradient(input_data, teaching_signal, responses, lr=1, loss_type='hinge'):

    # FORMAT IS REQUIRED TO BE B, N, D where b is a single batch, n is the number of classes, d is the dimension

    if loss_type == 'hinge':

        teaching_signal[teaching_signal == 0] = -1
        mask = (responses * teaching_signal) > 1
        target_gradient = -1 * input_data * teaching_signal * -1 * lr
        raw_gradient = target_gradient.clone()
        target_gradient[mask] = 0
        target_gradient = target_gradient.squeeze()

        out = dict(gradient=target_gradient, gradient_no_mask=raw_gradient.squeeze())
        return out

    elif loss_type == 'cross_entropy':

        y_hat = torch.nn.functional.softmax(responses, dim=-2)
        q = y_hat - teaching_signal
        grad = input_data * q
        grad = grad * -1 * lr
        out = dict(gradient=grad)
        return out


def prepare_tensors_for_gradient_computations(data, labels, hyperplanes, num_classes, loss_type='hinge'):

    assert len(data.shape) == 2
    num_points, dim = data.shape
    assert labels.shape == (num_points, )
    assert hyperplanes.shape[-2:] == (num_classes, dim)

    hyperplanes = hyperplanes.flatten(0, -3)
    bsz, _, _ = hyperplanes.shape
    responses = torch.einsum('bcd, pd -> bpc', hyperplanes, data)
    responses = responses.unsqueeze(-1).repeat(1, 1, 1, dim)
    input_data = data.reshape(1, num_points, 1, dim).repeat(bsz, 1, num_classes, 1)

    if loss_type == 'hinge':
        one_hot_labels = ut.datasets.preprocessing.one_hot(labels, num_classes,
                                                           replace_zeros_with_neg_one=True)
    else:
        one_hot_labels = ut.datasets.preprocessing.one_hot(labels, num_classes,
                                                           replace_zeros_with_neg_one=False)

    teaching_signal = one_hot_labels.reshape(1, num_points, num_classes, 1).repeat(bsz, 1, 1, dim)

    out = dict(teaching_signal=teaching_signal, responses=responses, input_data=input_data)
    return out


def generate_min_max_gradients(gradients):

    gradients_min = gradients.min(-3)[0]
    gradients_max = gradients.max(-3)[0]
    return gradients_min, gradients_max
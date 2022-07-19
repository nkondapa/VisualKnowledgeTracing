import numpy as np
import torch
from sklearn.metrics import top_k_accuracy_score
import utilities.compute_confidence_intervals as cci


def topk_response_prediction_accuracy(out, formatted_data):

    num_classes = out['responses'].shape[-1]
    class_scores = {}
    for k in range(1, num_classes):
        score = top_k_accuracy_score(torch.argmax(formatted_data['responses_t'], dim=-1).flatten(0, -1),
                             out['responses'].flatten(0, -2), k=k)

        class_scores[k] = score

    return class_scores


def compute_correctness_statistics(label_response_matrix, convolve=3):
    '''

    :param label_response_matrix: expects the shape of the matrix to be N, 2, T
    :param convolve:
    :return:
    '''
    data = label_response_matrix
    label_matrix = data[:, 0].astype(int)
    correctness = (data[:, 1] == label_matrix).astype(float)
    num_label_types = len(np.unique(label_matrix))
    correct_by_label_matrix = np.zeros((num_label_types, 2, correctness.shape[-1]))

    for i in range(label_matrix.shape[0]):
        for j in range(label_matrix.shape[1]):
            correct_by_label_matrix[label_matrix[i, j], 0, j] += correctness[i, j]
            correct_by_label_matrix[label_matrix[i, j], 1, j] += 1

    per_class_moving_average = []
    for i in range(num_label_types):
        num = np.convolve(correct_by_label_matrix[i, 0], np.ones(convolve), 'valid')
        den = np.convolve(correct_by_label_matrix[i, 1], np.ones(convolve), 'valid')
        per_class_moving_average.append([num, den])

    pcma = np.stack(per_class_moving_average)
    average = (data[:, 1] == label_matrix).astype(float).mean(0)
    return dict(per_class_moving_average=pcma, average=average, label_matrix=label_matrix)


def plot_per_class_moving_average(axes, pcma, num_label_types, compute_confidence=True, plot_params=None):
    moving_average = pcma[:, 0] / pcma[:, 1]

    if plot_params is None:
        plot_params = {}

    color = plot_params.get('color', None)
    label = plot_params.get('label', None)
    for i in range(num_label_types):
        if compute_confidence:
            l, m, h = cci.generate_ci_over_list(cci.propci_wilson_cc, pcma[i, 0], pcma[i, 1])
            axes[i].plot(np.arange(len(m)), m, color=color, label=label, marker='o')
            axes[i].fill_between(range(1, len(m) + 1), l, h, alpha=0.3, color=color)
        else:
            axes[i].plot(np.arange(moving_average.shape[1]), moving_average[i], color=color, label=label)
        axes[i].set_ylim([0, 1])

    return axes
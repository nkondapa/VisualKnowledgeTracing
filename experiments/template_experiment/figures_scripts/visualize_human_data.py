import torch
import experiments.vkt_experiment.experiment_utilities as eut
import matplotlib.pyplot as plt
from experiments.vkt_experiment.analysis_utilities.human_data_statistics import compute_correctness_statistics, plot_per_class_moving_average
import matplotlib
import config
import os

font = {'weight' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)

dataset_to_label_names = {
    'butterflies': ['Cabbage White', 'Monarch', 'Queen', 'Red Admiral', 'Viceroy'],
    'eyes': ['DME', 'Drusen', 'Normal'],
    'greebles': ['Agara', 'Bari', 'Cooka'],
}

dataset_list = ['butterflies', 'greebles', 'eyes']
for dataset in dataset_list:
    if dataset == 'butterflies':
        p = '/home/nkondapa/PycharmProjects/VisualKnowledgeTracing_private/experiments/vkt_experiment/experiment_learner_data/butterflies/butterflies_fold0_train_test/'
        num_classes = 5
    elif dataset == 'greebles':
        p = '/home/nkondapa/PycharmProjects/VisualKnowledgeTracing_private/experiments/vkt_experiment/experiment_learner_data/greebles/greebles_fold0_train_test/'
        num_classes = 3
    elif dataset == 'eyes':
        p = '/home/nkondapa/PycharmProjects/VisualKnowledgeTracing_private/experiments/vkt_experiment/experiment_learner_data/oct/OCT_fold0_train_test/'
        num_classes = 3
    else:
        raise Exception

    learner_data, experiment_vars = eut.load_learner_data(p, get_training_data=False, device='cpu')

    mask_dict = experiment_vars['mask_dict']
    gt_label = learner_data[:, :, mask_dict['gt_label']].argmax(dim=-1)
    responses = learner_data[:, :, mask_dict['class_response']].squeeze()

    lr_matrix = torch.stack([gt_label, responses]).permute(1, 0, 2).numpy()
    print(lr_matrix)
    compute_correctness_statistics(lr_matrix, convolve=1)

    convolve_sizes = [3]
    fig, axes = plt.subplots(len(convolve_sizes), num_classes, squeeze=False, constrained_layout=True)
    fig.set_size_inches(15, 3)

    for ci, convolve_size in enumerate(convolve_sizes):
        tmp = compute_correctness_statistics(lr_matrix, convolve=convolve_size)
        pcma = tmp['per_class_moving_average']
        average = tmp['average']
        plot_per_class_moving_average(axes[ci, :], pcma[:, :, :], num_classes, plot_params={'label': 'Human Data'})
        if ci == 0:
            for k, ax in enumerate(axes[ci]):
                ax.set_title(dataset_to_label_names[dataset][k])

        axes[ci, 0].set_ylabel(f'Average Probability')
        # axes[ci, 2].set_xlabel(f'Time Step')
        for j in range(num_classes):
            if j == 2:
                axes[ci, j].axvline(30, label='Train-Test Divider', color='red')
            if j > 0:
                axes[ci, j].set_yticklabels([])
            axes[ci, j].axvline(30, color='red')
            axes[ci, j].set_xlim((-2.2, 46.2))
            axes[ci, j].set_xticklabels([])
            print(axes[ci, j].get_ylim())
            # axes[ci, j].set_yticks([0.25, 0.5, 0.75, 1.00])

        axes[ci, 2].legend(fontsize=15)

    path = f'../figures/human_data_statistics/'
    os.makedirs(f'{path}', exist_ok=True)
    plt.savefig(f'{path}/{dataset}_average_probabilities.pdf')
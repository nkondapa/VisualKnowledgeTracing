import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import config


def pull_training_data_from_model_dict(experiment_group_name, experiment_name, experiment_save_specifier):

    path = f'{config.ROOT}/experiments/{experiment_group_name}/experiment_results/{experiment_name}/{experiment_save_specifier}/results.pkl'
    with open(path, 'rb') as f:
        results = pkl.load(f)

    train_acc = np.stack(results['train_per_sequence_per_learner_acc'])
    val_acc = np.stack(results['val_per_sequence_per_learner_acc'])
    test_acc = np.stack(results['test_per_sequence_per_learner_acc'])

    train_acc_per_epoch = train_acc.mean((1, 2, 3))
    val_acc_per_epoch = val_acc.mean((1, 2, 3))
    test_acc = test_acc.mean((1, 2, 3))

    train_loss = np.stack(results['train_per_epoch_loss'])
    val_loss = np.stack(results['val_per_epoch_loss'])
    test_loss = results['test_loss']

    out = dict(
        train_acc_per_epoch=train_acc_per_epoch, train_loss_per_epoch=train_loss,
        val_acc_per_epoch=val_acc_per_epoch, val_loss_per_epoch=val_loss,
        test_acc=test_acc, test_loss=test_loss,
    )

    return out


def plot_curves(ax, data_dict, plot_params=None):
    if plot_params is None:
        plot_params = {'metric': 'loss', 'splits': ['train']}

    smoothing = plot_params.get('smoothing', 0.05)
    metric = plot_params['metric']
    splits = plot_params['splits']
    for split in splits:
        data = data_dict[f'{split}_{metric}_per_epoch']
        convolve_size = int(smoothing * len(data))
        if convolve_size > 1:
            data = np.convolve(data, np.ones(convolve_size) / convolve_size, mode='valid')
        ax.plot(data, label=split)

    ax.legend()

    if 'title' in plot_params:
        ax.set_title(plot_params['title'])

    ax.set_xlabel('epochs')
    ax.set_ylabel(metric)
    ax.set_yscale(plot_params.get('scale', 'log'))

    if 'val' in splits:
        if metric == 'acc':
            x_pos = data.argmax()
        else:
            x_pos = data.argmin()

        ax.axvline(x_pos, color='red')
        xt = ax.get_xticks()
        xlims = ax.get_xlim()
        xt = np.append(xt, x_pos)
        ax.set_xticks(xt)
        ax.set_xlim(*xlims)

    # plt.suptitle()
import matplotlib.pyplot as plt
import numpy as np


def plot_label_prob_dict(axes, label_prob_dict, mode='per_learner_per_timestep', **plot_params):

    if mode == 'per_learner_per_timestep':

        for li, label in enumerate(label_prob_dict):
            pp = {'marker': 'o'}

            if 'k_list' in plot_params:
                for k in plot_params['k_list']:
                    axes[li].plot(label_prob_dict[label]['mean_per_learner_per_timestep'][k, :].T, **pp, label=k)
            else:
                if 'legend_labels' in plot_params:
                    pp['label'] = plot_params['legend_labels'][li]
                axes[li].plot(label_prob_dict[label]['mean_per_learner_per_timestep'].T, **pp)
            axes[li].set_ylim([0, 1])

    elif mode == 'per_timestep':
        for li, label in enumerate(label_prob_dict):
            pp = {'marker': 'o'}
            if 'legend_labels' in plot_params:
                pp['label'] = plot_params['legend_labels'][li]
            if 'color' in plot_params:
                pp['color'] = plot_params['color']
            mean = label_prob_dict[label]['mean_per_timestep']
            std = label_prob_dict[label]['std_per_timestep']
            handle = axes[li].plot(mean, **pp)
            color = handle[0].get_color()
            axes[li].fill_between(range(len(mean)), mean - std, mean + std, alpha=0.3, color=color)
            axes[li].set_ylim([0, 1])


def plot_probability_delta_matrix(delta_dict):

    bin_size = 5
    d = delta_dict[bin_size]
    num_classes = len(d.keys())
    tmp = generate_matrices_from_delta_dict(d)
    class_mat = tmp['class_mat']
    binary_mat = tmp['binary_mat']

    fig, axes = plt.subplots(1, num_classes)
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                axes[i].plot(class_mat[i, j], label=j)
            else:
                axes[i].plot(class_mat[i, j], linestyle='--', label=j)

        axes[i].legend()
        axes[i].set_title(f'Class {i}')
    axes[0].set_xlabel('Time bin')
    axes[0].set_ylabel(f'Average $\Delta P$, {bin_size} binned timesteps')
    plt.suptitle('Average $\Delta P(r_t=c\ |\ y_{t}=c, y_{t-1}=c, r_{t-1})$')

    fig, axes = plt.subplots(1, num_classes)
    for i in range(num_classes):
        axes[i].plot(binary_mat[i, 1], linestyle='--', marker='x', label='incorrect')
        axes[i].plot(binary_mat[i, 0], label='correct')

        axes[i].legend()
        axes[i].set_title(f'Class {i}')
    axes[0].set_xlabel('Time bin')
    axes[0].set_ylabel(f'Average $\Delta P$, {bin_size} binned timesteps')
    plt.suptitle('Average $\Delta P(r_t=c\ |\ y_{t}=c, y_{t-1}=c, r_{t-1}=c)$')

    bin_size = 30
    d = delta_dict[bin_size]
    num_classes = len(d.keys())
    tmp = generate_matrices_from_delta_dict(d)
    class_mat = tmp['class_mat']
    binary_mat = tmp['binary_mat']
    plt.figure()
    plt.imshow(class_mat)
    plt.colorbar()


def generate_matrices_from_delta_dict(d):
    _k = list(d.keys())[0]
    num_time_bins = len(d[_k][_k]['average_delta'])
    num_classes = len(d.keys())
    class_mat = np.zeros(shape=(num_classes, num_classes, num_time_bins))
    binary_mat = np.zeros(shape=(num_classes, 2, num_time_bins))
    # print(delta_dict)
    for k1i, k1 in enumerate(d.keys()):
        for k2i, k2 in enumerate(d[k1].keys()):
            if k2i < class_mat.shape[1]:
                # print(k1i, k2i, d[k1][k2])
                class_mat[k1i, k2i] = d[k1][k2]['average_delta']
            else:
                binary_mat[k1i, 1] = d[k1][k2]['average_delta']
            if k1i == k2i:
                binary_mat[k1i, 0] = d[k1][k2]['average_delta']

    return dict(class_mat=class_mat, binary_mat=binary_mat)
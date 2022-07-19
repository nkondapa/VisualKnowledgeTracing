import numpy as np
import torch


def generate_label_probs_dict(labels, probs):
    # convert probs into mean probs for each label, per learner, per timestep
    # expects probs to be bsz, seq_len, num_queries, num_classes
    label_probs_dict = {}
    for li, label in enumerate(np.unique(labels)):
        mask = label == labels
        label_probs = probs[:, :, mask, li]
        label_probs_dict[li] = {
            'mean_per_learner_per_timestep': label_probs.mean((-1)).numpy(),
            'std_per_learner_per_timestep': label_probs.std((-1)).numpy(),
            'mean_per_timestep': label_probs.mean((0, -1)).numpy(),
            'std_per_timestep': label_probs.std((0, -1)).numpy()
        }

    return label_probs_dict


def collect_prob_deltas(label_prob_dict, responses_t, teaching_signal_t, time_bin_fractions=None, **kwargs):
    seq_len = teaching_signal_t.shape[1]
    num_classes = teaching_signal_t.shape[-1]

    if time_bin_fractions is None:
        time_bin_fractions = [1, 5 / 30]
    if type(time_bin_fractions) is not list:
        time_bin_fractions = [time_bin_fractions]

    assert min(time_bin_fractions) > 0
    assert max(time_bin_fractions) <= 1

    gt_label = teaching_signal_t.argmax(-1).cpu().numpy()
    responses = responses_t.argmax(-1).squeeze().cpu().numpy()

    delta_dict = {}
    for ci in range(num_classes):
        delta = label_prob_dict[ci]['mean'][:, 1:] - label_prob_dict[ci]['mean'][:, :-1]
        gtc_mask = gt_label == ci
        for cj in range(num_classes):
            rc_mask = responses == cj
            delta_mask = (gtc_mask[:, :-1] & rc_mask[:, :-1])
            for time_bin_fraction in time_bin_fractions:
                bin_size = max(1, int(time_bin_fraction * seq_len))

                if bin_size not in delta_dict:
                    delta_dict[bin_size] = {}
                if ci not in delta_dict[bin_size]:
                    delta_dict[bin_size][ci] = {}

                averages = []
                counts = []
                for bi in range(int(1 / time_bin_fraction)):
                    si = bi * bin_size
                    ei = min((bi + 1) * bin_size, seq_len)
                    sub_mask = delta_mask[:, si:ei]
                    delta_slice = delta[:, si:ei][sub_mask]
                    if len(delta_slice) != 0:
                        _delta_slice_sum = delta_slice.sum()
                        _sub_mask_count = sub_mask.sum().item()
                        averages.append(_delta_slice_sum / _sub_mask_count)
                        counts.append(_sub_mask_count)
                    else:
                        averages.append(np.nan)
                        counts.append(np.nan)

                delta_dict[bin_size][ci][cj] = {'average_delta': np.array(averages),
                                                'count': np.array(counts)}

    for bi, bin_size in enumerate(delta_dict.keys()):
        for ci in range(num_classes):
            incorrect_numerator = None
            incorrect_denominator = None
            for cj in range(num_classes):
                if ci != cj:
                    entry = delta_dict[bin_size][ci][cj].copy()
                    entry['average_delta'][np.isnan(entry['average_delta'])] = 0
                    entry['count'][np.isnan(entry['count'])] = 0
                    if incorrect_numerator is None:
                        incorrect_numerator = entry['average_delta'] * entry['count']
                        incorrect_denominator = entry['count']
                    else:
                        incorrect_numerator = incorrect_numerator + entry['average_delta'] * entry['count']
                        incorrect_denominator = incorrect_denominator + entry['count']
            delta_dict[bin_size][ci]['incorrect'] = {'average_delta': incorrect_numerator / incorrect_denominator,
                                                     'count': incorrect_denominator}

    return delta_dict
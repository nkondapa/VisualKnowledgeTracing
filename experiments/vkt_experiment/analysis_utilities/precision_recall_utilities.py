from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
import torch
import pandas as pd
from experiments.vkt_experiment.analysis_utilities import names


def extract_relevant_data_from_model_output(out, formatted_data_split, experiment_vars, subset_mask=None):
    ### This function should handle model type (if model types have different properties)
    if subset_mask is None:
        subset_mask = torch.ones(formatted_data_split['responses_t'].shape[0], dtype=torch.bool)
    if 'probs' not in out:
        probs = torch.softmax(out['responses'].detach(), dim=-1)[subset_mask]
        flat_probs = probs.flatten(0, -2).cpu().numpy()
        targets = formatted_data_split['responses_t'][subset_mask].flatten(0, -2).cpu().numpy()
        return dict(probs=probs, model_probs=flat_probs,
                    target_responses=targets, num_classes=experiment_vars.num_classes)

    else:
        # FOR DKT_TRANSLATION MODEL
        probs = out['probs'][subset_mask]
        flat_probs = probs.flatten(0, -2).cpu().numpy()
        targets = formatted_data_split['responses_t'][subset_mask].flatten(0, -2).cpu().numpy()
        return dict(probs=probs, model_probs=flat_probs, dkt_variant=True, qid=formatted_data_split['qid'],
                    target_responses=targets, num_classes=experiment_vars.num_classes, num_samples=subset_mask.sum())


def compute_precision_recall_per_class(model_probs, target_responses, num_classes, **kwargs):
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    thresholds = dict()

    for i in range(num_classes):
        precision[i], recall[i], thresholds[i] = precision_recall_curve(target_responses[:, i], model_probs[:, i])
        average_precision[i] = average_precision_score(target_responses[:, i], model_probs[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], thresholds['micro'] = precision_recall_curve(
        target_responses.ravel(), model_probs.ravel()
    )
    average_precision["micro"] = average_precision_score(target_responses, model_probs, average="micro")
    average_precision["macro"] = average_precision_score(target_responses, model_probs, average="macro")

    return dict(recall=recall, precision=precision, average_precision=average_precision, thresholds=thresholds)


def compute_precision_recall_per_class_dkt_variant(model_probs, target_responses, num_classes, qid, **kwargs):
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    thresholds = dict()

    for i in range(num_classes):
        precision[i], recall[i], thresholds[i] = precision_recall_curve(target_responses[:, i], model_probs[:, i])
        average_precision[i] = average_precision_score(target_responses[:, i], model_probs[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], thresholds['micro'] = precision_recall_curve(
        target_responses.ravel(), model_probs.ravel()
    )
    average_precision["micro"] = average_precision_score(target_responses, model_probs, average="micro")
    average_precision["macro"] = average_precision_score(target_responses, model_probs, average="macro")

    return dict(recall=recall, precision=precision, average_precision=average_precision, thresholds=thresholds)


def plot_micro_average(ax, recall, precision, average_precision, **kwargs):
    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot(ax)

    return display.ax_


def plot_precision_recall_per_class(ax, recall, precision, average_precision, num_classes, **kwargs):
    # setup plot details
    colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

    # _, ax = plt.subplots(figsize=(7, 8))

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = ax.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        ax.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

    for i, color in zip(range(num_classes), colors):
        display = PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i],
        )
        display.plot(ax=ax, name=f"Precision-recall for class {i}", color=color)

    # add the legend for the iso-f1 curves
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["iso-f1 curves"])
    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_title("Extension of Precision-Recall curve to multi-class")


def compute_pr(model_out, formatted_data, experiment_vars, subset_mask=None):
    pru_dict = extract_relevant_data_from_model_output(model_out, formatted_data, experiment_vars, subset_mask=subset_mask)
    if 'dkt_variant' in pru_dict:
        pr_dict = compute_precision_recall_per_class_dkt_variant(**pru_dict)
    else:
        pr_dict = compute_precision_recall_per_class(**pru_dict)

    return pr_dict


def convert_metric_summary_dict_into_multilevel_df_for_precision_recall(metric_summary_dict, target_keys=None):

    if type(metric_summary_dict) is dict:
        df = pd.DataFrame.from_dict(metric_summary_dict, orient='index')
    else:
        df = metric_summary_dict

    if target_keys is None:
        target_keys = ['script_name', 'dataset']

    multilevel_dict = {}
    for row_id in df.index:
        entry = df.loc[row_id]
        key1 = entry[target_keys[0]]
        key2 = entry[target_keys[1]]
        if key1 not in multilevel_dict:
            multilevel_dict[key1] = {}

        multilevel_dict[key1][key2] = {'train_seq': entry['train_seq_AP']}
        if 'test_seq_AP' in entry:
            multilevel_dict[key1][key2]['test_seq'] = entry['test_seq_AP']

    new_dict = {}
    for k1, d1 in multilevel_dict.items():  # script level
        print(k1)
        if target_keys[0] == 'script_name':
            k1 = names.script_name_to_descriptive_model_name.get(k1, k1)
        for k2, d2 in d1.items():      # dataset level

            dl_dict = None
            if 'butterflies' in k2:
                dl_dict = names.dataset_label_names['butterflies']
            if 'greebles' in k2:
                dl_dict = names.dataset_label_names['greebles']
            if 'sld' in k2:
                dl_dict = None

            for k3, d3 in d2.items():  # train vs test seq level
                if type(d3) is dict:   # sld does not have test seq
                    for k4, d4 in d3.items():  # individual sub p-r analysis
                        if dl_dict is not None and k4 in dl_dict:
                            k4 = dl_dict[k4]
                        if k1 not in new_dict:
                            new_dict[k1] = {(k2, k3, k4): d4}
                        else:
                            new_dict[k1][(k2, k3, k4)] = d4
                else:
                    if k1 not in new_dict:
                        new_dict[k1] = {(k2, k3): d3}
                    else:
                        new_dict[k1][(k2, k3)] = d3

    new_df = pd.DataFrame.from_dict(new_dict, orient='index')

    return new_df




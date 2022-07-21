import experiments.vkt_experiment.experiment_utilities as eut
import pandas as pd
import torch


def prep_data_for_metrics(model, learner_data, mask_dict, split_dict, training_data, evars, device=None):
    target_dict = dict(learner_data=learner_data)
    target_dict['input_data_dict'] = eut.format_input_data(
        target_dict['learner_data'], mask_dict, evars.sequence_length, **training_data)
    target_dict['formatted_data'] = model.format_data(**target_dict['input_data_dict'],
                                                      mask_dict=mask_dict, split_dict=split_dict,
                                                      learner_data=learner_data, device=device)

    for split in target_dict['formatted_data']:
        sd = target_dict['formatted_data'][split]
        sd['responses_t'].argmax(-1)
        tmp = (sd['responses_t'].argmax(-1).squeeze(-1) == sd['teaching_signal_t'].argmax(-1))
        num_correct_per_class = []
        for i in range(sd['teaching_signal_t'].shape[-1]):
            sub_tmp = (sd['teaching_signal_t'].argmax(-1) == i) & tmp
            num_correct_per_class.append(sub_tmp.type(torch.int).sum(1))

        sd['num_correct_per_learner'] = tmp.type(torch.int).sum(1)
        sd['num_correct_per_learner_per_class'] = torch.stack(num_correct_per_class)

    return target_dict


def combine_splits(formatted_data, splits):

    tmp = None
    for split in splits:
        if tmp is None:
            tmp = formatted_data[split].copy()
        else:
            for key in tmp:
                tmp[key] = torch.cat([tmp[key], formatted_data[split][key]], dim=0)

    return tmp
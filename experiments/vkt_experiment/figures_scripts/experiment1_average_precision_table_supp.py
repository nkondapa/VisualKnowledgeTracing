import os
import numpy as np

from experiments.vkt_experiment.analysis_utilities import precision_recall_utilities as pru
import pickle as pkl

tag_name = 'experiment1'
path = f'../experiment_analysis_output/{tag_name}/'
out_path = f'../figures/{tag_name}/supp_paper_tables/'
os.makedirs(path, exist_ok=True)
os.makedirs(out_path, exist_ok=True)

with open(path + 'metric_summary_dict_pr.pkl', 'rb') as file:
    metric_summary_dict = pkl.load(file)

script_names = [
    'runner_gt_baseline',
    'runner_baseline',
    'runner_timestep_baseline',
    'runner_dkt_translation_model',
    'runner_direct_response_transformer',
    'runner_prototype_baseline',
    'runner_exemplar_baseline',
    'runner_direct_response_model1v1',
    'runner_direct_response_model1v2',
    'runner_direct_response_model1v3',
    'runner_classifier_model1v1',
    'runner_classifier_model1v2',
    'runner_classifier_model1v3',

]

script_name_to_paper_name = {
    'runner_gt_baseline': 'GT Label$\dagger$',
    'runner_baseline': '$\phi_{static}\dagger$',
    'runner_timestep_baseline': '$\phi_{static\_time}\dagger$',
    'runner_dkt_translation_model': '$\phi_{dkt}$',
    'runner_direct_response_model1v1': '$\phi_{direct(base)}$',
    'runner_direct_response_model1v2': '$\phi_{direct(y)}$',
    'runner_direct_response_model1v3': '$\phi_{direct(y, \mathbf{z})}\dagger$',
    'runner_classifier_model1v1': '$\phi_{cls\_pred(base)}$',
    'runner_classifier_model1v2': '$\phi_{cls\_pred(y)}\dagger$',
    'runner_classifier_model1v3': '$\phi_{cls\_pred(y, \mathbf{z})}$',
    'runner_direct_response_transformer': '$\phi_{transformer}$',
    'runner_prototype_baseline': '$\phi_{prototype}$',
    'runner_exemplar_baseline': '$\phi_{exemplar}$',
}


average_dict = {}
for ki, k in enumerate(metric_summary_dict):
    entry = metric_summary_dict[k]
    script_name = entry['script_name']
    dataset_name = entry['dataset']
    dataset_name = dataset_name[:dataset_name.find('_fold')]
    if entry['script_name'] not in average_dict:
        average_dict[script_name] = {'script_name': script_name}
        average_dict[script_name]['dataset_names'] = []
    if dataset_name not in average_dict[script_name]:
        average_dict[script_name]['dataset_names'].append(dataset_name)
        average_dict[script_name][dataset_name] = {}
        average_dict[script_name][dataset_name]['train_seq_AP'] = {}
        average_dict[script_name][dataset_name]['test_seq_AP'] = {}

    for key in entry['train_seq_AP']:
        if key not in average_dict[script_name][dataset_name]['train_seq_AP']:
            average_dict[script_name][dataset_name]['train_seq_AP'][key] = []
        average_dict[script_name][dataset_name]['train_seq_AP'][key].append(entry['train_seq_AP'][key])

    for key in entry['test_seq_AP']:
        if key not in average_dict[script_name][dataset_name]['test_seq_AP']:
            average_dict[script_name][dataset_name]['test_seq_AP'][key] = []
        average_dict[script_name][dataset_name]['test_seq_AP'][key].append(entry['test_seq_AP'][key])

for script_name in average_dict:
    for dataset_name in average_dict[script_name]['dataset_names']:
        entry = average_dict[script_name][dataset_name]
        print(script_name, dataset_name)
        for kk in entry['train_seq_AP']:
            mean = np.mean(entry['train_seq_AP'][kk])
            std = np.std(entry['train_seq_AP'][kk])
            entry['train_seq_AP'][kk] = f'{mean:0.2f}$\pm${std:0.2f}'
        for kk in entry['test_seq_AP']:
            mean = np.mean(entry['test_seq_AP'][kk])
            std = np.std(entry['test_seq_AP'][kk])
            entry['test_seq_AP'][kk] = f'{mean:0.2f}$\pm${std:0.2f}'

refactor_entries = {}
c = 0
for script_name in script_names:
    for dataset_name in average_dict[script_name]['dataset_names']:
        paper_name = script_name_to_paper_name[script_name]
        entry = average_dict[script_name][dataset_name]
        for key in list(entry['train_seq_AP'].keys()):
            if key != 'macro' and key != 'micro':
                # del entry['train_seq_AP'][key]
                del entry['test_seq_AP'][key]

        refactor_entries[c] = {'script_name': paper_name, 'dataset': dataset_name,
                               'train_seq_AP': entry['train_seq_AP'], 'test_seq_AP': entry['test_seq_AP']}
        c += 1
pr_df = pru.convert_metric_summary_dict_into_multilevel_df_for_precision_recall(refactor_entries)
p = f'{out_path}/html_files/'
os.makedirs(p, exist_ok=True)
for level_id in pr_df.columns.unique(0):
    s = pr_df[[level_id]]. \
        style.set_properties(**{'border': '1.3px solid black'}). \
        set_table_styles(
        [{'selector': 'th', 'props': [('font-size', '12pt'), ('border-style', 'solid'), ('border-width', '1px')]}]). \
        to_html(f'{p}/{tag_name}_average_precision_{level_id}.html')
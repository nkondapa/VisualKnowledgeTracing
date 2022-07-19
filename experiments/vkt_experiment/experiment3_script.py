import sys
import os
import config
import pickle as pkl
import time
import utilities
from datetime import datetime
import traceback
import sys
import numpy as np
import subprocess

'''
GOAL Compare models on the selected datasets, 6000 seconds
Datasets
 - butterflies (human)
 - greebles
 - butterflies (synthetic)

'''

start_time = time.time()
dt_object = datetime.fromtimestamp(start_time)
rng = np.random.default_rng(367354245)
seed = rng.integers(0, 93423501230)
tag_name = sys.argv[0].split('/')[-1].replace('_script.py', '')
try:

    print(os.path.abspath(sys.argv[0]))
    experiment_group_name = os.path.abspath(sys.argv[0]).split('/')[-2]

    kfold = 5
    dataset_list = []
    experiment_data_names = []
    for fi in range(kfold):
        experiment_data_names.extend(['butterflies'])
        dataset_list.extend(
            [f'butterflies_fold{fi}']
        )

    feature_dimensions = [8, 16, 32, 64]
    script_names = [
        'runner_direct_response_model1v3',
    ]

    information_level = 'argmax'
    for script_name in script_names:
        for d, dataset in enumerate(dataset_list):
            for feature_dimension in feature_dimensions:
                specific_start_time = time.time()
                try:
                    lr = None
                    fe_lr = None
                    experiment_data_name = experiment_data_names[d]
                    experiment_data_specifier = dataset
                    batch_size = 16
                    hidden_size = 128

                    ip_lr = 0.0001  # no longer used

                    if script_name == 'runner_baseline':
                        lr = 0.0001
                        fe_lr = 0.0001
                    elif script_name == 'runner_gt_baseline':
                        lr = 0.0001
                        fe_lr = 0.0001
                    elif script_name == 'runner_timestep_baseline':
                        lr = 0.0001
                        fe_lr = 0.0001
                    elif script_name == 'runner_dkt_translation_model':
                        lr = 0.001
                        fe_lr = 0.0001
                    elif 'runner_classifier_model' in script_name or 'runner_direct_response_model' in script_name:
                        lr = 0.001
                        fe_lr = 0.00001

                    print('-' * 89)
                    print(experiment_data_name, script_name)
                    epochs = 400
                    model_name = script_name.replace('runner_', '')
                    ess = f'/{tag_name}/model={model_name}_dataset={dataset.replace("/", "_")}_embdim={feature_dimension}'

                    lstm_model_specific_str = ''
                    if 'baseline' not in script_name:
                        lstm_model_specific_str = f' --hidden-state-dimension {hidden_size} --ip-lr {ip_lr}' \
                                                  f' --label-smoothing 0.0' \
                                                  f' --num-layers 3 --dropout 0.0'
                    os.system(
                        f'python {config.ROOT}/experiments/{experiment_group_name}/{script_name}.py'
                        f' --experiment-data-specifier {experiment_data_specifier}'
                        f' --experiment-save-specifier {ess}'
                        f' --experiment-group-name {experiment_group_name}'
                        f' --experiment-data-name {experiment_data_name}'
                        f' --experiment-name {model_name}'
                        f' --log-interval {2} --batch-size {batch_size} --epochs {epochs} --cuda --seed {seed}'
                        f' --information-level {information_level}'
                        f' --feature-dimension {feature_dimension} --early-stopping'
                        f' --lr {lr} --fe-lr {fe_lr}' + lstm_model_specific_str
                    )

                except Exception or TypeError as e:
                    print(e)

except Exception as e:
    print(e)


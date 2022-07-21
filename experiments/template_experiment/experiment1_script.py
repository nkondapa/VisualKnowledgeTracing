import os
import config
import time
from datetime import datetime
import sys
import numpy as np
from experiment_utilities import load_learning_rates

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

# This is by default going to strip script.py from the name of the file and use the rest as the experiment tag
tag_name = sys.argv[0].split('/')[-1].replace('_script.py', '')
try:

    print(os.path.abspath(sys.argv[0]))
    experiment_group_name = os.path.abspath(sys.argv[0]).split('/')[-2]

    # generates a list of datasets and the folds
    kfold = 5
    dataset_list = []
    experiment_data_names = []
    for fi in range(kfold):
        experiment_data_names.extend(['butterflies', 'greebles', 'oct'])
        dataset_list.extend(
            [f'butterflies_fold{fi}', f'greebles_fold{fi}', f'OCT_fold{fi}']
        )

    # Set which models you wish to run
    script_names = [
        'runner_baseline',
        'runner_timestep_baseline',
        'runner_gt_baseline',
        'runner_dkt_translation_model',
        'runner_classifier_model1v1',
        'runner_classifier_model1v2',
        'runner_classifier_model1v3',
        'runner_direct_response_model1v1',
        'runner_direct_response_model1v2',
        'runner_direct_response_model1v3',
        'runner_direct_response_transformer',
        'runner_prototype_baseline',
        'runner_exemplar_baseline',
    ]

    # iterates through the datasets and scripts and runs them all
    information_level = 'argmax'
    for script_name in script_names:
        for d, dataset in enumerate(dataset_list):
            specific_start_time = time.time()
            try:
                experiment_data_name = experiment_data_names[d]
                experiment_data_specifier = dataset

                # set parameters
                feature_dimension = 16
                batch_size = 16
                hidden_size = 128

                # parameter sweeps were conducted to select good learning rates for each model type
                lr, fe_lr = load_learning_rates(script_name)

                print('-' * 89)
                print(experiment_data_name, script_name)
                epochs = 400
                model_name = script_name.replace('runner_', '')

                # TODO - make sure all the parameters being varied are included in the filename
                #  (so that the model doesn't overwrite another one)
                ess = f'/{tag_name}/model={model_name}_dataset={dataset.replace("/", "_")}'


                # lstm specific parameters
                lstm_model_specific_str = ''
                if 'baseline' not in script_name:
                    lstm_model_specific_str = f' --hidden-state-dimension {hidden_size}' \
                                              f' --label-smoothing 0.00' \
                                              f' --num-layers 3 --dropout 0.0'

                # equivalent to running the runner script from terminal and passsing these args
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


import sys
from utilities.database_utilities import load_mongodb_data, save, convert_user_data_to_tensor_kfold
import os
import config
from experiment_utilities import experiment_group_name


folder = experiment_group_name
data_source = 'OCT'
exp_name = 'OCT'
db_dict = load_mongodb_data(f'{config.VKT_DATASET_FOLDER}/', data_source)
valid_session_params = {'num_correct_test_threshold': 4, 'num_hits_completed_threshold': 45}
learner_data, experiment_vars = convert_user_data_to_tensor_kfold(db_dict, 3, exp_name, valid_session_params,
                                                                  split_num_folds=(21, 4, 5),
                                                                  kfold=5, seed=765, return_train_and_test=True)

for i, evar in enumerate(experiment_vars):
    experiment_name_ = exp_name + '_fold' + str(i)
    print(evar['input_params']['sequence_length'])
    save(learner_data, evar, '', 'oct/' + f'{experiment_name_}_train_test')
    train_seq_len = evar['input_params']['train_sequence_length']
    evar['input_params']['sequence_length'] = train_seq_len
    save(learner_data[:, :train_seq_len], evar, '', 'oct/' + f'{experiment_name_}')


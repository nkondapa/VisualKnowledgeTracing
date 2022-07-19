import numpy as np
import pickle as pkl
import pandas as pd
import copy
import json
import datetime
import os
import torch
import torchvision
from PIL import Image
import config
import random


def load_dataset(db_name, **kwargs):
    dataset_root = f'{config.VKT_DATASET_FOLDER}/{db_name}/images/'

    with open(f'{config.VKT_DATASET_FOLDER}/{db_name}/{db_name}.json', 'r') as f:
        exp_data = json.load(f)
        image_urls = np.array(exp_data['image_urls'])
        labels = torch.LongTensor(exp_data['labels'])
        class_names = exp_data['class_names']

    im_paths = list(map(lambda x: dataset_root + x.split('/')[-1], image_urls))
    if db_name == 'OCT':
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size=(144, 144)),
                                                   torchvision.transforms.ToTensor()])
    else:
        transform = torchvision.transforms.ToTensor()

    img_tensors = []
    for im_path in im_paths:
        img_tensors.append(transform(Image.open(im_path)))

    data = torch.stack(img_tensors)
    if data.shape[1] == 4:
        data = data[:, :-1]

    return dict(data=data, labels=labels, class_names=class_names)


def load_mongodb_data(folder, db_name):

    with open(f'{folder}/{db_name}/mongodb_data/db_dict.json', 'r') as f:
        db_dict = json.load(f)

    return db_dict


def extract_data(db_dict, image_label_dict, response_rank_index=0, valid_params=None):

    if valid_params is None:
        valid_params = {}

    data = []
    image_ids = []

    for username in db_dict:
        session = db_dict[username]
        if is_session_valid(session, **valid_params):
            session_data = []
            session_image_ids = []
            for hit in session['hit_list']:
                session_image_ids.append(hit['HIT_image_ids'])
                label = image_label_dict[hit['HIT_image_ids']]
                response = int(json.loads(hit['response'])[response_rank_index])
                session_data.append((label, response))

            data.append(np.array(list(map(list, zip(*session_data)))))
            image_ids.append(session_image_ids)

    label_response_matrix = np.stack(data)
    return dict(label_response_matrix=label_response_matrix, image_ids=image_ids)


def compute_correctness_statistics(label_response_matrix, convolve=3):
    data = label_response_matrix
    label_matrix = data[:, 0].astype(int)
    correctness = (data[:, 1] == label_matrix).astype(float)
    num_label_types = len(np.unique(label_matrix))
    correct_by_label_matrix = np.zeros((num_label_types, 2, correctness.shape[-1]))

    for i in range(label_matrix.shape[0]):
        for j in range(label_matrix.shape[1]):
            correct_by_label_matrix[label_matrix[i, j], 0, j] += correctness[i, j]
            correct_by_label_matrix[label_matrix[i, j], 1, j] += 1

    per_class_moving_average = []
    for i in range(num_label_types):
        num = np.convolve(correct_by_label_matrix[i, 0], np.ones(convolve), 'valid')
        den = np.convolve(correct_by_label_matrix[i, 1], np.ones(convolve), 'valid')
        per_class_moving_average.append([num, den])

    pcma = np.stack(per_class_moving_average)
    average = (data[:, 1] == label_matrix).astype(float).mean(0)
    return dict(per_class_moving_average=pcma, average=average, label_matrix=label_matrix)


def format_valid_session_params(valid_session_params):
    '''
        Leave the params in a json-friendly format until they are going to be used.
    '''
    vsp = copy.deepcopy(valid_session_params)
    if valid_session_params['submission_database'] is not None:
        data = pd.read_csv(valid_session_params['submission_database'])
        vsp['submission_database'] = data

    return vsp


def is_session_valid(session, **kwargs):
    username = session['username']

    if 'submission_database' in kwargs:
        sub_db = kwargs['submission_database']
        if (username == sub_db.participant_id).any():
            mask = session['username'] == sub_db.participant_id
            status = sub_db[mask]['status'].item()
            if status != 'APPROVED' and status != 'AWAITING REVIEW':
                print(f'Skipping user {username} due to status {status}!')
                return False
        else:
            return False

    if 'num_correct_test_threshold' in kwargs:
        if session['num_correct_test'] < kwargs['num_correct_test_threshold']:
            print(f'Skipping user {username} due to {session["num_correct_test"]} test correct below threshold!')
            return False

    if 'num_correct_upper_bound' in kwargs:
        if session['num_correct_test'] >= kwargs['num_correct_upper_bound']:
            print(f'Skipping user {username} due to {session["num_correct_test"]} test correct above bound!')
            return False

    if 'num_hits_completed_threshold' in kwargs:
        if session['hits_completed'] < kwargs['num_hits_completed_threshold']:
            print(f'Skipping user {username} due to {session["hits_completed"]} test below threshold!')
            return False

    if 'time_spent_threshold' in kwargs:
        start_time = datetime.datetime.strptime(session['timestamp'], '%Y-%m-%d %H:%M:%S.%f')
        end_time = datetime.datetime.strptime(session['session_finish_time'], '%Y-%m-%d %H:%M:%S.%f')
        td = end_time - start_time
        mins = td.seconds / 60
        if mins < kwargs['time_spent_threshold']:
            print(f'Skipping user {username} due to {mins} time spent below threshold!')

    print(f'user {username} passes checks!')
    return True


def save(learner_data, experiment_vars, path_to_experiment_folder, experiment_name):
    # Save data
    path = os.path.join(path_to_experiment_folder, 'experiment_learner_data', experiment_name + '/')
    os.makedirs(path, exist_ok=True)
    torch.save(learner_data, path + 'learner_data.pth')

    # Save mask_dict, data_tuple_dict, etc...
    with open(path + 'experiment_vars.pkl', 'wb') as f:
        pkl.dump(experiment_vars, f)


################ FORMATTING UTILITIES ################

def compute_flattened_data_length(data_tuple_dict):

    flat_data_length = 0
    for key in data_tuple_dict:
        flat_data_length += np.prod(data_tuple_dict[key])

    mask_dict = {}
    curr_len = 0
    for key in data_tuple_dict:
        tmp = torch.zeros(flat_data_length).type(torch.bool)
        data_len = np.prod(data_tuple_dict[key])
        tmp[curr_len:(curr_len + data_len)] = True
        mask_dict[key] = tmp
        curr_len += data_len

    mask_dict = mask_dict

    return mask_dict, flat_data_length


def convert_user_data_to_tensor(user_data, num_classes, exp_name, valid_session_params,
                                split_fractions=(0.75, 0.12, 0.13), return_train_and_test=False):
    assert sum(split_fractions) == 1
    num_supervised_queries = 1
    num_unsupervised_queries = 0
    num_queries = num_supervised_queries + num_unsupervised_queries
    data_tuple_dict = {
        'supervised_ind': (num_supervised_queries, ),
        'gt_label': (num_classes,),
        'rank_response': (num_classes, ),
        'class_response': (1, ),
    }

    mask_dict, flat_data_length = compute_flattened_data_length(data_tuple_dict)
    user_data = list(user_data.values())
    random.shuffle(user_data)
    total_num_users = len(user_data)
    sequence_length = user_data[0]['num_hits_per_session']
    learner_data = torch.zeros(total_num_users, sequence_length, flat_data_length)
    include_mask = torch.ones(total_num_users, dtype=torch.bool)
    testing_start_num = None

    butterflies_shuffle = f'{config.VKT_DATASET_FOLDER}/{exp_name}/{exp_name}.json'
    with open(butterflies_shuffle, 'r') as f:
        butterflies_exp_data = json.load(f)
        image_urls = np.array(butterflies_exp_data['image_urls'])

    if exp_name == 'greebles':
        id_dict = dict(zip(image_urls, map(lambda x: int(x.split('/')[-1].replace('.png', '')), image_urls)))
    elif exp_name == 'OCT':
        id_dict = dict(zip(image_urls, map(lambda x: int(x.split('/')[-1].replace('.jpeg', '')), image_urls)))
    else:
        id_dict = dict(zip(image_urls, np.arange(len(image_urls))))

    for k, u in enumerate(user_data):
        if not is_session_valid(u, **valid_session_params):
            include_mask[k] = False
            continue
        image_ids = torch.FloatTensor(list(map(lambda x: id_dict[x], image_urls[u['shuffled_indices']]))).unsqueeze(1)
        gt_labels = []
        responses = []
        class_responses = []
        for t in range(sequence_length):
            if testing_start_num is None:
                if u['hit_list'][t]['feedback'] == 'testing':
                    testing_start_num = t
            tmp = torch.zeros(num_classes)
            tmp[u['hit_list'][t]['target_label']] = 1
            gt_labels.append(tmp)
            ranking = torch.zeros(num_classes)
            rank_ordered_response_tensor = torch.LongTensor(list(map(int, json.loads(u['hit_list'][t]['response']))))
            ranking[rank_ordered_response_tensor] = torch.arange(num_classes, 0, -1).type(torch.float)
            responses.append(ranking)
            class_responses.append(torch.argmax(ranking))
        gt_labels = torch.stack(gt_labels)
        responses = torch.stack(responses)
        class_responses = torch.stack(class_responses).unsqueeze(-1)

        vec = torch.cat([image_ids, gt_labels, responses, class_responses], dim=1)
        learner_data[k, :] = vec

    is_train_mask = torch.zeros(sequence_length, dtype=torch.bool)
    is_train_mask[:testing_start_num] = True
    learner_data = learner_data[include_mask]

    total_num_users = learner_data.shape[0]
    num_learners = [int(total_num_users * split_fractions[i]) for i in range(len(split_fractions))]
    if sum(num_learners) < total_num_users:
        num_learners[0] += (total_num_users - sum(num_learners))

    split_dict = {'train': np.arange(0, num_learners[0]),
                  'val': np.arange(num_learners[0], sum(num_learners[:2])),
                  'test': np.arange(sum(num_learners[:2]), sum(num_learners[:3]))}

    if not return_train_and_test:
        learner_data = learner_data[:, is_train_mask]
        sequence_length = sum(is_train_mask)

    dataset_params = {'add_bias': False, 'load_function': load_dataset}
    input_params = dict(dataset_name=exp_name, dataset_params=dataset_params,
                        num_learners=num_learners,
                        sequence_length=sequence_length,
                        train_sequence_length=sum(is_train_mask),
                        num_supervised_queries=num_supervised_queries,
                        num_unsupervised_queries=num_unsupervised_queries,
                        num_queries=num_queries,
                        num_classes=num_classes)

    experiment_vars = {
        'split_dict': split_dict,
        'mask_dict': mask_dict,
        'input_params': input_params,
        'data_tuple_dict': data_tuple_dict,
    }
    return learner_data, experiment_vars


def convert_user_data_to_tensor_kfold(user_data, num_classes, exp_name, valid_session_params,
                                      split_num_folds=(21, 4, 5), kfold=5, seed=1, return_train_and_test=False):
    num_supervised_queries = 1
    num_unsupervised_queries = 0
    num_queries = num_supervised_queries + num_unsupervised_queries
    data_tuple_dict = {
        'supervised_ind': (num_supervised_queries, ),
        'gt_label': (num_classes,),
        'rank_response': (num_classes, ),
        'class_response': (1, ),
    }

    rng = np.random.default_rng(seed=seed)
    mask_dict, flat_data_length = compute_flattened_data_length(data_tuple_dict)
    user_data = list(user_data.values())
    random.shuffle(user_data)
    total_num_users = len(user_data)
    sequence_length = user_data[0]['num_hits_per_session']
    learner_data = torch.zeros(total_num_users, sequence_length, flat_data_length)
    include_mask = torch.ones(total_num_users, dtype=torch.bool)
    testing_start_num = None

    data_shuffle = f'{config.VKT_DATASET_FOLDER}/{exp_name}/{exp_name}.json'
    with open(data_shuffle, 'r') as f:
        exp_data = json.load(f)
        image_urls = np.array(exp_data['image_urls'])

    for k, u in enumerate(user_data):
        if not is_session_valid(u, **valid_session_params):
            include_mask[k] = False
            continue

        image_inds = torch.FloatTensor(u['shuffled_indices']).unsqueeze(1)
        gt_labels = []
        responses = []
        class_responses = []
        for t in range(sequence_length):
            if testing_start_num is None:
                if u['hit_list'][t]['feedback'] == 'testing':
                    testing_start_num = t
            tmp = torch.zeros(num_classes)
            tmp[u['hit_list'][t]['target_label']] = 1
            gt_labels.append(tmp)
            ranking = torch.zeros(num_classes)
            rank_ordered_response_tensor = torch.LongTensor(list(map(int, json.loads(u['hit_list'][t]['response']))))
            ranking[rank_ordered_response_tensor] = torch.arange(num_classes, 0, -1).type(torch.float)
            responses.append(ranking)
            class_responses.append(rank_ordered_response_tensor[0].type(torch.float))
        gt_labels = torch.stack(gt_labels)
        responses = torch.stack(responses)
        class_responses = torch.stack(class_responses).unsqueeze(-1)

        vec = torch.cat([image_inds, gt_labels, responses, class_responses], dim=1)
        learner_data[k, :] = vec

    is_train_mask = torch.zeros(sequence_length, dtype=torch.bool)
    is_train_mask[:testing_start_num] = True
    learner_data = learner_data[include_mask]

    # Assign number of users per fold (deals with remainders)
    total_num_users = learner_data.shape[0]
    fold_sizes = np.array([int(total_num_users / sum(split_num_folds))] * sum(split_num_folds))
    remainder = total_num_users - fold_sizes.sum()
    tmp = np.arange(sum(split_num_folds))
    fold_sizes[rng.choice(tmp, remainder, replace=False)] += 1

    # Assign indices for users according to the fold sizes
    user_indices_per_fold = []
    tmp_arr = np.concatenate([np.array([0]), np.cumsum(fold_sizes)])
    for t in range(1, len(tmp_arr)):
        user_indices_per_fold.append(np.arange(tmp_arr[t - 1], tmp_arr[t]))

    user_indices_per_fold = np.array(user_indices_per_fold, dtype=object)

    # if total num folds == 10, below matrix is generated;
    # we now can pick k rows and split by train, val, test along columns and get a random split + no overlap
    # array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
    #        [2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
    #        [3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
    #        [4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
    #        [5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
    #        [6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
    #        [7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
    #        [8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
    #        [9, 0, 1, 2, 3, 4, 5, 6, 7, 8]])
    fold_index_matrix = (np.arange(0, sum(split_num_folds)).reshape(-1, 1).repeat(sum(split_num_folds), 1)
                         + np.arange(0, sum(split_num_folds))) % sum(split_num_folds)

    # split columns into train, val (optional), test
    if len(split_num_folds) == 2:
        train_fold_cols = fold_index_matrix[:, :split_num_folds[0]]
        val_fold_cols = fold_index_matrix[:, split_num_folds[0]:]
        test_fold_cols = fold_index_matrix[:, split_num_folds[0]:]
    else:
        split_partition = np.cumsum(split_num_folds)
        train_fold_cols = fold_index_matrix[:, :split_partition[0]]
        val_fold_cols = fold_index_matrix[:, split_partition[0]:split_partition[1]]
        test_fold_cols = fold_index_matrix[:, split_partition[1]:]

    # pick k random rows in the fold index matrix
    fold_rows = rng.choice(np.arange(sum(split_num_folds)), kfold, replace=False)

    print(fold_rows)
    cross_val_split_dicts = []
    for k in range(kfold):

        # expand fold indices back into the user indices for each fold
        fold_train_user_inds = np.concatenate([user_indices_per_fold[j] for j in train_fold_cols[fold_rows[k]]]).astype(np.int)
        fold_val_user_inds = np.concatenate([user_indices_per_fold[j] for j in val_fold_cols[fold_rows[k]]]).astype(np.int)
        fold_test_user_inds = np.concatenate([user_indices_per_fold[j] for j in test_fold_cols[fold_rows[k]]]).astype(np.int)
        cross_val_split_dicts.append(
            {'train': fold_train_user_inds,
             'val': fold_val_user_inds,
             'test': fold_test_user_inds}
        )
        # #### verify
        # num_users_per_split = len(fold_train_user_inds), len(fold_val_user_inds), len(fold_test_user_inds)
        # print(set(fold_train_user_inds).intersection(set(fold_test_user_inds)))
        # print(set(fold_train_user_inds).intersection(set(fold_val_user_inds)))
        #
        # if len(split_num_folds) == 3:
        #     print(set(fold_val_user_inds).intersection(set(fold_test_user_inds)))
        #     print(num_users_per_split, sum(num_users_per_split), total_num_users)
        # else:
        #     print(num_users_per_split[0], num_users_per_split[2], sum(num_users_per_split) - num_users_per_split[1], total_num_users)

    if not return_train_and_test:
        learner_data = learner_data[:, is_train_mask]
        sequence_length = sum(is_train_mask)

    dataset_params = {'add_bias': False, 'load_function': load_dataset, 'db_name': exp_name}
    input_params = dict(dataset_name=exp_name, dataset_params=dataset_params,
                        sequence_length=sequence_length,
                        train_sequence_length=sum(is_train_mask).item(),
                        num_supervised_queries=num_supervised_queries,
                        num_unsupervised_queries=num_unsupervised_queries,
                        num_queries=num_queries,
                        num_classes=num_classes)

    experiment_vars = [
        {
            'split_dict': cross_val_split_dicts[k],
            'mask_dict': copy.copy(mask_dict),
            'input_params': copy.copy(input_params),
            'data_tuple_dict': copy.copy(data_tuple_dict),
        } for k in range(kfold)]

    return learner_data, experiment_vars
from utilities.custom_losses import *
import utilities
import numpy as np


def generate_loss_function(loss_function_class_handle, **params):

    loss_function = loss_function_class_handle(**params)
    return loss_function


def get_loss_function_label_transform(loss_function_class_handle):

    if loss_function_class_handle == torch.nn.CrossEntropyLoss:
        def transform(x, **kwargs):
            return x
    elif loss_function_class_handle == HingeLoss:
        def transform(x, num_classes, **kwargs):
            return utilities.datasets.preprocessing.one_hot(x, num_classes, replace_zeros_with_neg_one=True)
    else:
        return ValueError(f'{loss_function_class_handle} is unknown!')

    return transform


def generate_optimizer(optimizer_class_handle, **params):

    optimizer = optimizer_class_handle(**params)
    return optimizer


def convert_responses_to_information_level(learner_data, mask_dict, num_classes, level='score', **kwargs):
    ## TARGET OUTPUT SHAPE is num learner, sequence length + 1, 1, num_classes

    if 'response' not in mask_dict and 'rank_response' not in mask_dict and 'class_response' in mask_dict and level != 'argmax':
        raise Exception('Responses of this type must use information level \'argmax\'')

    if level == 'argmax':
        responses = utilities.datasets.preprocessing.one_hot(learner_data[:, :, mask_dict['class_response']]
                                                      .type(torch.long), num_classes=num_classes)
    elif level == 'uncertain_argmax':
        responses = utilities.datasets.preprocessing.one_hot(learner_data[:, :, mask_dict['class_response']]
                                                      .type(torch.long), num_classes=num_classes)
    elif level == 'human_ranked':
        responses = learner_data[:, :, mask_dict['rank_response']].unsqueeze(-2)
    elif level == 'ranking':
        responses = learner_data[:, :, mask_dict['response']].argsort(-1).argsort(-1).type(torch.float).unsqueeze(-2)
    elif level == 'softmax_score':
        responses = torch.nn.functional.softmax(learner_data[:, :, mask_dict['response']].unsqueeze(2), -1)
    elif level == 'noisy_score':
        responses = learner_data[:, :, mask_dict['response']].unsqueeze(2)
        responses = responses + torch.FloatTensor(np.random.rand(*responses.shape)).to(responses.device) * kwargs['std_dev_scale'] * responses[:, :-1].std()
    elif level == 'noisy_softmax_score':
        responses = learner_data[:, :, mask_dict['response']].unsqueeze(2)
        responses = responses + torch.FloatTensor(np.random.rand(*responses.shape)).to(responses.device) * kwargs['std_dev_scale'] * responses[:, :-1].std()
        responses = torch.nn.functional.softmax(responses, -1)
    elif level == 'score':
        responses = learner_data[:, :, mask_dict['response']].unsqueeze(2)
    else:
        raise Exception(f'Information level {level} not supported!')

    return responses


def get_model_loss_function_for_information_level(level='score', **kwargs):

    if level == 'argmax':
        loss_function = CrossEntropyWrapper()
    elif level == 'uncertain_argmax':
        print(kwargs.get('label_smoothing'))
        loss_function = CrossEntropyWrapper(label_smoothing=kwargs.get('label_smoothing', 0.3))
    elif level == 'softmax_score':
        loss_function = softmax_mse_wrapper
    elif level == 'noisy_score':
        loss_function = torch.nn.MSELoss(reduction='none')
    elif level == 'noisy_softmax_score':
        loss_function = softmax_mse_wrapper
    elif level == 'score':
        loss_function = torch.nn.MSELoss(reduction='none')
    else:
        raise Exception(f'Information level {level} not supported!')

    return loss_function
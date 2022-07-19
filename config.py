import torch


SUBROOT = f'/home/nkondapa/'
ROOT = f'/home/nkondapa/PycharmProjects/VisualKnowledgeTracing/'
# SYNTHETIC_DATASET_FOLDER = f'{ROOT}/datasets/'
TRAINED_MODELS_FOLDER = f'{ROOT}/trained_models/'
VKT_DATASET_FOLDER = f'{SUBROOT}/Datasets/VKT_Datasets/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
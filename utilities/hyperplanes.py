import numpy as np
import torch
from config import device
import gc
from sklearn.metrics.pairwise import cosine_distances


def compute_slope_intercept(point1, point2):
    if point2[0] == point1[0]:
        slope = np.inf
        intercept = 0 if point1[0] == 0 else None
        return slope, intercept

    slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
    intercept = _compute_intercept(slope, point1)
    return slope, intercept


def _compute_intercept(slope, point):
    intercept = point[1] - slope * point[0]
    return intercept


def compute_midpoint(point1, point2):
    midpt = (point1 + point2) / 2
    return midpt


def compute_hyperplane_weights(slope, point):
    if slope == 0:
        intercept = point[1]
        return np.array([slope, -1, intercept])

    # @TODO generalize for dimensions > 2
    intercept = _compute_intercept(slope, point)
    # x2 = m * x1 + b -> 0 = m * x1 -1 * x2 + b
    return torch.FloatTensor([slope, -1, intercept])


def compute_separating_hyperplane_from_points(p1, p2):

    w = p1 - p2
    w[-1] = (p1 - p2) @ ((p1 + p2) / 2)
    w = w / torch.norm(w)

    return w


def unit_normalize(hyperplane_matrix):
    return hyperplane_matrix / np.linalg.norm(hyperplane_matrix, axis=1)[:, None]


def distance_between_batched_hyperplane_matrices(hyperplane_matrix_batch1, hyperplane_matrix_batch2, params):
    distance_mode = params.get('distance_mode')

    if distance_mode == 'classification':
        data = params['classification_data']
        if 'subsample_data' in params:
            data1 = data[params['subsample_data']]

        num_learners, seq_len, num_classes, dim = hyperplane_matrix_batch1.shape

        response1 = torch.einsum('kscd, pd->kscp', torch.FloatTensor(hyperplane_matrix_batch1).to(device),
                             torch.FloatTensor(data).to(device)).cpu()
        torch.cuda.empty_cache()
        response2 = torch.einsum('kscd, pd->kscp', torch.FloatTensor(hyperplane_matrix_batch2).to(device),
                             torch.FloatTensor(data).to(device)).cpu()

        torch.cuda.empty_cache()
        pred1 = torch.argmax(response1, dim=-2)
        pred2 = torch.argmax(response2, dim=-2)
        diff = (pred1 != pred2)
        diff_float = diff.type(torch.FloatTensor)
        distances = diff_float.mean(-1).numpy()

        gt_hm = params.get('return_per_class', 0)
        keys = list(range(num_classes))
        if gt_hm == 0:
            kn, vn = torch.unique(pred1[diff], return_counts=True)
            kd, vd = torch.unique(pred1, return_counts=True)
            dn = dict(zip(kn.numpy(), vn.numpy()))
            num_comparisons_per_class = dict(zip(kd.numpy(), vd.numpy()))
            per_class_error = dict([(k, dn.get(k, 0) / num_comparisons_per_class.get(k, -1)) for k in keys])
        elif gt_hm == 1:
            kn, vn = torch.unique(pred2[diff], return_counts=True)
            kd, vd = torch.unique(pred2, return_counts=True)
            dn = dict(zip(kn.numpy(), vn.numpy()))
            num_comparisons_per_class = dict(zip(kd.numpy(), vd.numpy()))
            per_class_error = dict([(k, dn.get(k, 0) / num_comparisons_per_class.get(k, -1)) for k in keys])
        else:
            raise Exception()
        results = dict(distances=distances, per_class_error=per_class_error,
                       num_comparisons_per_class=num_comparisons_per_class)
        return results

    if distance_mode == 'classification2':
        if 'classification_data1' in params and 'classification_data2' in params:
            data1 = params['classification_data1']
            data2 = params['classification_data2']
            if 'subsample_data' in params:
                data1 = data1[params['subsample_data']]
                data2 = data2[params['subsample_data']]

            orig_shape = hyperplane_matrix_batch1.shape
            num_classes, dim = orig_shape[-2:]
            hyperplane_matrix_batch1 = hyperplane_matrix_batch1.reshape(-1, num_classes, dim)
            hyperplane_matrix_batch2 = hyperplane_matrix_batch2.reshape(-1, num_classes, dim)

            if type(hyperplane_matrix_batch1) == np.ndarray:
                hyperplane_matrix_batch1 = torch.FloatTensor(hyperplane_matrix_batch1)
            if type(hyperplane_matrix_batch2) == np.ndarray:
                hyperplane_matrix_batch2 = torch.FloatTensor(hyperplane_matrix_batch2)

            if type(data1) == np.ndarray:
                data1 = torch.FloatTensor(data1)
            if type(data2) == np.ndarray:
                data2 = torch.FloatTensor(data2)

            response1 = torch.einsum('bcd, pd->bcp', hyperplane_matrix_batch1.to(device), data1.to(device)).cpu()
            torch.cuda.empty_cache()
            response2 = torch.einsum('bcd, pd->bcp', hyperplane_matrix_batch2.to(device), data2.to(device)).cpu()
            torch.cuda.empty_cache()
            pred1 = torch.argmax(response1, dim=-2)
            pred2 = torch.argmax(response2, dim=-2)
            diff = (pred1 != pred2)
            diff_float = diff.type(torch.float)
            distances = diff_float.mean(-1).numpy()

            gt_hm = params.get('return_per_class', 0)
            keys = list(range(num_classes))
            if gt_hm == 0:
                kn, vn = torch.unique(pred1[diff], return_counts=True)
                kd, vd = torch.unique(pred1, return_counts=True)
                dn = dict(zip(kn.numpy(), vn.numpy()))
                num_comparisons_per_class = dict(zip(kd.numpy(), vd.numpy()))
                per_class_error = dict([(k, dn.get(k, 0) / num_comparisons_per_class.get(k, -1)) for k in keys])
            elif gt_hm == 1:
                kn, vn = torch.unique(pred2[diff], return_counts=True)
                kd, vd = torch.unique(pred2, return_counts=True)
                dn = dict(zip(kn.numpy(), vn.numpy()))
                num_comparisons_per_class = dict(zip(kd.numpy(), vd.numpy()))
                per_class_error = dict([(k, dn.get(k, 0) / num_comparisons_per_class.get(k, -1)) for k in keys])
            else:
                raise Exception()

            results = dict(distances=distances, per_class_error=per_class_error,
                           num_comparisons_per_class=num_comparisons_per_class)
            return results
        else:
            raise ValueError('Missing classification data.')


def distance_between_hyperplane_matrices(hyperplane_matrix1, hyperplane_matrix2, params):
    num_classes, dim = hyperplane_matrix1.shape
    distance_mode = params.get('distance', 'euclidean')
    if distance_mode == 'euclidean':
        distances = np.linalg.norm(unit_normalize(hyperplane_matrix1) - unit_normalize(hyperplane_matrix2), axis=1)
        return distances.squeeze()
    elif distance_mode == 'cosine':
        distances = 1 - torch.nn.functional.cosine_similarity(torch.FloatTensor(hyperplane_matrix1),
                                                              torch.FloatTensor(hyperplane_matrix2), dim=-1).numpy()
        return distances
    elif distance_mode == 'classification':
        if 'classification_data' in params:
            data = params['classification_data']
            pred1 = np.argmax(data @ hyperplane_matrix1.T, axis=1)
            pred2 = np.argmax(data @ hyperplane_matrix2.T, axis=1)
            diff = pred1 != pred2
            distance = np.mean(np.array(diff))

            if params.get('return_per_class', None) is not None:
                gt_hm = params['return_per_class']
                keys = list(range(num_classes))
                if gt_hm == 0:
                    kn, vn = np.unique(pred1[diff], return_counts=True)
                    kd, vd = np.unique(pred1, return_counts=True)
                    dn = dict(zip(kn, vn))
                    dd = dict(zip(kd, vd))
                    d = dict([(k, dn.get(k, 0) / dd.get(k, -1)) for k in keys])
                elif gt_hm == 1:
                    kn, vn = np.unique(pred2[diff], return_counts=True)
                    kd, vd = np.unique(pred2, return_counts=True)
                    dn = dict(zip(kn, vn))
                    dd = dict(zip(kd, vd))
                    d = dict([(k, dn.get(k, 0) / dd.get(k, np.nan)) for k in keys])
                else:
                    raise Exception()
                return distance, d, dd

            return distance
        else:
            raise ValueError('Missing classification data.')
    elif distance_mode == 'classification2':
        if 'classification_data1' in params and 'classification_data2' in params:
            data1 = params['classification_data1']
            data2 = params['classification_data2']
            pred1 = np.argmax(data1 @ hyperplane_matrix1.T, axis=1)
            pred2 = np.argmax(data2 @ hyperplane_matrix2.T, axis=1)
            diff = pred1 != pred2
            distance = np.mean(np.array(diff))

            if params.get('return_per_class', None) is not None:
                gt_hm = params['return_per_class']
                keys = list(range(num_classes))
                if gt_hm == 0:
                    kn, vn = np.unique(pred1[diff], return_counts=True)
                    kd, vd = np.unique(pred1, return_counts=True)
                    dn = dict(zip(kn, vn))
                    dd = dict(zip(kd, vd))
                    d = dict([(k, dn.get(k, 0) / dd.get(k, -1)) for k in keys])
                elif gt_hm == 1:
                    kn, vn = np.unique(pred2[diff], return_counts=True)
                    kd, vd = np.unique(pred2, return_counts=True)
                    dn = dict(zip(kn, vn))
                    dd = dict(zip(kd, vd))
                    d = dict([(k, dn.get(k, 0) / dd.get(k, np.nan)) for k in keys])
                else:
                    raise Exception()
                return distance, d, dd

            return distance
        else:
            raise ValueError('Missing classification data.')
    else:
        raise ValueError(f'{distance_mode} not recognized!')


def generate_gridded_hyperplanes_from_batch(hyperplane_batch, dim_pairs=None, classes=None, grid_size=20):

    if dim_pairs is None:
        dim_pairs = [(0, 1)]

    print()
    hyperplane_batch = hyperplane_batch.flatten(0, -3)
    b, c, d = hyperplane_batch.shape

    if classes is None:
        classes = torch.arange(0, c, 1)

    mins = hyperplane_batch.min(0)[0]
    maxes = hyperplane_batch.max(0)[0]
    ranges = (maxes - mins)
    centers = ranges / 2

    scale = 2
    ranges *= scale
    maxes = centers + (ranges / 2)
    mins = centers - (ranges / 2)
    step_sizes = ranges/(grid_size - 1)

    grid_tensors = torch.zeros(size=(*mins.shape, grid_size))
    for ci in range(c):
        for di in range(d):
            # print(torch.arange(mins[ci, di], maxes[ci, di] + 0.001, step_sizes[ci, di]).shape)
            grid_tensors[ci, di] = torch.arange(mins[ci, di], maxes[ci, di] + 0.001, step_sizes[ci, di])

    import matplotlib.pyplot as plt
    meshes = torch.zeros(size=(len(classes), len(dim_pairs), grid_size ** 2, c, d))
    for cli, cl in enumerate(classes):
        for dpi, dim_pair in enumerate(dim_pairs):
            dim1, dim2 = dim_pair
            g1, g2 = torch.meshgrid(grid_tensors[cl, dim1], grid_tensors[cl, dim2])
            mesh_list = torch.zeros(size=(grid_size, grid_size, c, d))
            not_dim_pair = list(set(range(d)).difference(dim_pair))
            not_cl = list(set(range(c)).difference([cl.item()]))
            mesh_list[:, :, cl, not_dim_pair] = centers[cl, not_dim_pair].reshape(1, 1, -1).repeat(grid_size, grid_size, 1).cpu()
            mesh_list[:, :, not_cl, :] = centers[not_cl].reshape(1, 1, len(not_cl), d).repeat(grid_size, grid_size, 1, 1).cpu()

            mesh_list[:, :, cl, dim1] = g1
            mesh_list[:, :, cl, dim2] = g2

            meshes[cl, dpi] = mesh_list.flatten(0, -3)

    return meshes
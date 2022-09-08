import torch
import numpy as np
import sys

from utility.config import base_path, max_cac_val

sys.path.insert(0, base_path + '/src')


def pre_process_label(mean, std, labels):
    cac_score_clip = np.clip(labels.detach().cpu(), a_min=0, a_max=max_cac_val)
    log_cac_score = np.log(cac_score_clip + 0.001)
    return (log_cac_score - mean) / std


def mean_std_cac_score_log(loader):
    cac_score = torch.cat([labels for (_, labels) in loader]).numpy()
    cac_score_clip = np.clip(cac_score, a_min=0, a_max=max_cac_val)
    log_cac_score = np.log(cac_score_clip + 0.001)
    return log_cac_score.mean(), log_cac_score.std()


def local_copy_str_kfold(dataset):
    data = [dataset[j][0] for j in range(len(dataset))]
    label = [dataset[j][1] for j in range(len(dataset))]
    return data, label


def to_class(continuos_values, labels, th):
    classes_labels = [0 if labels[i] <= th else 1 for i in range(labels.size(dim=0))]
    output_labels = [0 if continuos_values[i] <= th else 1 for i in range(continuos_values.size(dim=0))]
    return torch.tensor(output_labels), torch.tensor(classes_labels)
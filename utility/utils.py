from pip import main
import torch
import collections
import random 
import os

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import seaborn as sns
import pandas as pd 

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)


def get_transforms(img_size, crop, mean, std):
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    return train_transforms, test_transform



def local_copy(dataset, require_cac_score):
    if require_cac_score:
        return  [(dataset[j][0],dataset[j][1],dataset[j][2]) for j in range(len(dataset))]
    else:
        return [(dataset[j][0],dataset[j][1]) for j in range(len(dataset))]


## Visualization - cross validation utils

def save_losses_fold(train_losses, test_losses, best_test_acc, fold, path_plot):
    plt.figure(figsize=(16, 8))
    plt.title(f'Best accuracy : {best_test_acc:.4f}')
    plt.plot(train_losses, label='Train loss')
    plt.plot(test_losses, label='Test loss')
    plt.legend()
    plt.savefig(path_plot  + 'losses_fold' + str(fold) + '.png')
    plt.close()


def save_cm_fold(true_labels, best_pred_labels, fold, path_plot):
    cm = confusion_matrix(true_labels, best_pred_labels)
    ax = sns.heatmap(cm, annot=True, fmt="d")
    hm = ax.get_figure()
    hm.savefig(path_plot + 'cm_fold' + str(fold) + '.png')
    hm.clf()
    plt.close(hm)


def save_roc_curve_fold(true_labels, probs, fold, path_plot):
    fpr, tpr, _ = roc_curve(true_labels, probs, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure(1)
    lw = 2
    plt.plot(fpr, tpr, color="darkorange", label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(path_plot  + 'fold_' + str(fold) + 'roc.png')
    plt.close()

    return roc_auc


def show_distribution(dataloader, set, fold, path_plot):
    batch_labels = [label.tolist() for _, label, _ in dataloader]
    label_flat_list = [item for sublist in batch_labels for item in sublist]
    count_labels = collections.OrderedDict(sorted(collections.Counter(label_flat_list).items()))
    
    val_samplesize = pd.DataFrame.from_dict(
    {'[0:10]': [count_labels[0]], 
     '> 10': count_labels[1],
    })

    sns.barplot(data=val_samplesize)
    plt.savefig(path_plot + str(set) + '_fold' + str(fold) + '.png')
    plt.close()


def save_metric(train, test, metric, fold, path_plot):
    plt.figure(figsize=(16, 8))
    plt.plot(train, label='Train ' + str(metric))
    plt.plot(test, label='Test ' + str(metric))
    plt.legend()
    plt.savefig(path_plot + str(metric) + 'fold_' + str(fold) + '_.png')
    plt.close()


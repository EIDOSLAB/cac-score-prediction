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


def probs_prediction(best_probs, pred_labels, true_labels, fold, PATH_PLOT):
    pred_and_labels = list(np.array((pred_labels, true_labels)).T)
    corret_preds, wrong_preds = [], []
    #print(f'True labels {true_labels} Pred Labels {pred_labels}')
    #print(f'Labels and Labels {pred_and_labels}')

    for index, (label, prediction) in enumerate(pred_and_labels):
        if label == prediction:
            corret_preds.append(best_probs[index])
        else:
            wrong_preds.append(best_probs[index])

    corret_preds = np.array(corret_preds)
    wrong_preds = np.array(wrong_preds)

    samples_for_corret_preds = samples_for_preds(np.array(corret_preds))
    samples_for_wrong_preds = samples_for_preds(np.array(wrong_preds))

    x = np.array(["> 0.5", "> 0.6", "> 0.7", "> 0.8", "> 0.9"])

    plt.bar(x, samples_for_corret_preds)
    plt.title('Probability of correct classified')
    plt.xlabel('Probability')
    plt.ylabel('Sample')
    plt.show()
    plt.savefig(PATH_PLOT  + 'probs_correct_clf_fold' + str(fold) + '.png')
    plt.close()

    plt.bar(x, samples_for_wrong_preds)
    plt.title('Probability of wrong classified')
    plt.xlabel('Probability')
    plt.ylabel('Sample')
    plt.show()
    plt.savefig(PATH_PLOT  + 'probs_wrong_clf_fold' + str(fold) + '.png')
    plt.close()




def analysis_results(best_probs, cac_scores, pred_labels, true_labels, fold, PATH_PLOT):
    pred_and_labels = list(np.array((pred_labels, true_labels)).T)
    cac_prob_wrong_preds = []
    cac_scores = np.clip(cac_scores, 0, 500)

    for index, (label, prediction) in enumerate(pred_and_labels):
        if label != prediction:
            cac_prob_wrong_preds.append((cac_scores[index] , best_probs[index]))

    cac_prob_wrong_preds.sort(key=lambda cac_prob: cac_prob[0])

    plt.title('Probability of sample wrong classified sort by CAC score')
    plt.scatter(*zip(*cac_prob_wrong_preds))
    plt.axvline(x = 10, color='r', label='Threshold')
    plt.xlabel('CAC score')
    plt.ylabel('Probability')
    plt.show()
    plt.savefig(PATH_PLOT  + 'analysis_results_' + str(fold) + '.png')
    plt.close()


def samples_for_preds(probs):
    samples_for_preds = []
    samples_for_preds.append(((0.5 < probs) & (probs <= 0.6)).sum())
    samples_for_preds.append(((0.6 < probs) & (probs <= 0.7)).sum())
    samples_for_preds.append(((0.7 < probs) & (probs <= 0.8)).sum())
    samples_for_preds.append(((0.8 < probs) & (probs <= 0.9)).sum())
    samples_for_preds.append(((0.8 < probs) & (probs <= 1)).sum())
    return samples_for_preds


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


def show_distribution_fold(dataloader, set, fold, path_plot):
    batch_labels = [label.tolist() for _, label in dataloader]
    label_flat_list = [item for sublist in batch_labels for item in sublist]
    count_labels = collections.OrderedDict(sorted(collections.Counter(label_flat_list).items()))
    
    val_samplesize = pd.DataFrame.from_dict(
    {'[0:10]': [count_labels[0]], 
     '> 10': count_labels[1],
    })

    sns.barplot(data=val_samplesize)
    plt.savefig(path_plot + str(set) + '_fold' + str(fold) + '.png')
    plt.close()
    print(f'For {set} Labels {count_labels}')


def save_metric_fold(train, test, metric, fold, path_plot):
    plt.figure(figsize=(16, 8))
    plt.plot(train, label='Train ' + str(metric))
    plt.plot(test, label='Test ' + str(metric))
    plt.legend()
    plt.savefig(path_plot + str(metric) + 'fold_' + str(fold) + '_.png')
    plt.close()


if __name__ == '__main__':
    max_probabilities =    [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
    best_model_cac_scores = [0, 10, 200, 400, 600, 800, 8000]
    best_pred_labels = [0, 0, 0, 0, 1, 1, 1]
    best_true_label =  [1, 1, 1, 1, 0, 0, 0]
    fold = 0
    PATH_PLOT = '/home/fiodice/project/plot_training/'

    analysis_results(max_probabilities, best_model_cac_scores, best_pred_labels, best_true_label, fold, PATH_PLOT)


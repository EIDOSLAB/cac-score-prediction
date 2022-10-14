import numpy as np
import matplotlib.pyplot as plt
import collections
import seaborn as sns
import pandas as pd 

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

from utility.config import TH_cac_score, base_path

PATH_PLOT = base_path + '/plot_training/'

plt.rc('font', size=13) 

############# Visualization error regression #############

def viz_cac_error_bins(labels, preds, mean, std, fold):
    labels = np.exp((labels * std) + mean - 0.001).flatten()
    preds = np.exp((preds * std) + mean - 0.001).flatten()

    labels_and_pred = list(np.array((labels, preds)).T)
    labels_and_pred.sort(key=lambda label: label[0])

    bins = [ 10, 500, 1000 , 2000]
    error, sample_for_bin = [], []
    error_on_bin, index_error, count = 0, 0, 0

    for label, prediction in labels_and_pred:
        count += 1
        error_on_bin += np.abs(label - prediction)
        if label > bins[index_error]:
            error.append(error_on_bin/count)
            sample_for_bin.append(count)
            error_on_bin = 0
            count = 0
            index_error += 1

    sample_for_bin.append(count)   
    error.append(error_on_bin/count)

    plt.figure(figsize=(14, 8))
    plt.ylabel("Mean error on bin")
    plt.xlabel("Calcium score")
    plt.bar([1, 2, 3, 4], height=error)
    plt.xticks([1, 2, 3, 4], [f'[0:10] N = {sample_for_bin[0]}',
                              f'[10:200] N = {sample_for_bin[1]}',
                              f'[200:500] N = {sample_for_bin[2]}',
                              f'[500:2000] N = {sample_for_bin[3]}'])
    plt.show()
    plt.savefig(PATH_PLOT  + 'errorbin_fold' + str(fold) + '.png')


def viz_cac_error(labels, preds, mean, std, fold, max_val=300, log_scale=False):
    if log_scale:
        labels = ((labels * std) + mean - 0.001).flatten()
        preds = ((preds * std) + mean - 0.001).flatten()
        th = (np.log(TH_cac_score + 0.001) - mean) / std
    else:
        labels = np.exp((labels * std) + mean - 0.001).flatten()
        preds = np.exp((preds * std) + mean - 0.001).flatten()
        labels = np.clip(labels,a_min=0, a_max=max_val).flatten()
        preds = np.clip(preds,a_min=0, a_max=max_val).flatten()
        th = TH_cac_score

    labels_and_preds = list(np.array((labels,preds)).T)
    labels_and_preds.sort(key=lambda label: label[0])

    top_error, bottom_error = [], []
    for label, prediction in labels_and_preds:
        top_error.append(prediction - label)
        bottom_error.append(0)

    plt.figure(figsize=(20, 8))
    plt.ylabel("Calcium score predicted")
    plt.xlabel("Calcium score label")
    plt.grid()
    plt.axhline(y = th, color = 'r', label = "Threshold")

    plt.errorbar(x = np.arange(start=0, stop=len(labels)), 
                 y = np.sort(labels), 
                 yerr=[bottom_error, top_error], fmt='o')
    plt.show()
    plt.savefig(PATH_PLOT  + 'error_cac_final_fold' + str(fold) + '.png')
    plt.close()


############# Visualization error classification #############

def viz_probs_prediction(best_probs, pred_labels, true_labels, fold):
    pred_and_labels = list(np.array((pred_labels, true_labels)).T)
    corret_preds, wrong_preds = [], []

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


def viz_samples_missclf(best_probs, cac_scores, pred_labels, true_labels, fold):
    pred_and_labels = list(np.array((pred_labels, true_labels)).T)
    cac_prob_wrong_preds = []
    cac_scores = np.clip(cac_scores, 0, 500)

    for index, (label, prediction) in enumerate(pred_and_labels):
        if label != prediction:
            cac_prob_wrong_preds.append((cac_scores[index] , best_probs[index]))

    cac_prob_wrong_preds.sort(key=lambda cac_prob: cac_prob[0])

    plt.figure(figsize=(10, 8))
    plt.title('Sample wrong classified sort by CAC score')
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
 
############# Utility distribution #############

def viz_distr_data(loader, fold, phase):
    scores = []
    for (_, labels) in loader:
        scores.append(labels.numpy()[0])
    plt.figure()
    plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
    plt.hist(scores, bins=int(180/1))
    plt.gca().set(title='Frequency Histogram CAC score fold ' + str(fold), xlabel='Calcium score', ylabel='Count')
    plt.savefig(PATH_PLOT + str(phase) + '_cac_frequency_fold_' + str(fold) + '.png')
    plt.close()


def viz_distr_data_binary(dataloader, set, fold):
    batch_labels = [label.tolist() for _, label, _ in dataloader]
    label_flat_list = [item for sublist in batch_labels for item in sublist]
    count_labels = collections.OrderedDict(sorted(collections.Counter(label_flat_list).items()))
    
    val_samplesize = pd.DataFrame.from_dict(
    {'[0:10]': [count_labels[0]], 
     '> 10': count_labels[1],
    })

    sns.barplot(data=val_samplesize)
    plt.savefig(PATH_PLOT + 'distr_' + str(set) + '_fold' + str(fold) + '.png')
    plt.close()


############# Utility save metrics #############

def save_metric(train, test, metric, fold):
    plt.figure(figsize=(16, 8))
    plt.plot(train, label='Train ' + str(metric))
    plt.plot(test, label='Test ' + str(metric))
    plt.legend()
    plt.savefig(PATH_PLOT + str(metric) + 'fold_' + str(fold) + '_.png')
    plt.close()


def save_cm(true_labels, best_pred_labels, fold):
    cm = confusion_matrix(true_labels, best_pred_labels)
    ax = sns.heatmap(cm, annot=True, fmt="d")
    hm = ax.get_figure()
    hm.savefig(PATH_PLOT + 'cm_fold' + str(fold) + '.png')
    hm.clf()
    plt.close(hm)


def save_roc_curve(true_labels, probs, fold):
    fpr, tpr, _ = roc_curve(true_labels, probs, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure(1)
    plt.plot(fpr, tpr, color="darkorange", label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(PATH_PLOT + 'roc_' + 'fold' + str(fold) + '.png')
    plt.close()

    return roc_auc


def save_losses(train_losses, test_losses, best_test_acc, fold):
    plt.figure(figsize=(16, 8))
    plt.title(f'Best accuracy : {best_test_acc:.4f}')
    plt.plot(train_losses, label='Train loss')
    plt.plot(test_losses, label='Test loss')
    plt.legend()
    plt.savefig(PATH_PLOT  + 'losses_fold' + str(fold) + '.png')
    plt.close()
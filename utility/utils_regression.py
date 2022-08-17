import torch
import numpy as np
import matplotlib.pyplot as plt

PATH_PLOT = '/home/fiodice/project/plot_training/'
MAX_CAC_VAL = 2000


def cac_prediction_error_bin(labels, preds, mean, std, fold):
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
    error.append(error_on_bin)

    print(f'EBins {error} size {len(error)}')

    plt.figure(figsize=(18, 10))
    plt.ylabel("Mean error on bin")
    plt.xlabel("Calcium score")
    plt.bar([1, 2, 3, 4], height=error)
    plt.xticks([1, 2, 3, 4], [f'[0:10] N = {sample_for_bin[0]}',
                              f'[10:200] N = {sample_for_bin[1]}',
                              f'[200:500] N = {sample_for_bin[2]}',
                              f'[500:2000] N = {sample_for_bin[3]}'])
    plt.show()
    plt.savefig(PATH_PLOT  + 'errorbin_fold' + str(fold) + '.png')


def cac_prediction_error(labels, preds, mean, std, fold, max_val=300, log_scale=False):
    th = 10
    if log_scale:
        labels = ((labels * std) + mean - 0.001).flatten()
        preds = ((preds * std) + mean - 0.001).flatten()
        th = (np.log(th + 0.001) - mean) / std
    else:
        labels = np.exp((labels * std) + mean - 0.001).flatten()
        preds = np.exp((preds * std) + mean - 0.001).flatten()
        labels = np.clip(labels,a_min=0, a_max=max_val).flatten()
        preds = np.clip(preds,a_min=0, a_max=max_val).flatten()

    labels_and_preds = list(np.array((labels,preds)).T)
    labels_and_preds.sort(key=lambda label: label[0])

    top_error, bottom_error = [], []
    for label, prediction in labels_and_preds:
        top_error.append(prediction - label)
        bottom_error.append(0)

    plt.figure(figsize=(18, 10))
    plt.xlabel("Samples")
    plt.ylabel("Calcium score predicted")
    plt.grid()
    plt.axhline(y = th, color = 'r', label = "Threshold")

    plt.errorbar(x = np.arange(start=0, stop=len(labels)), 
                 y = np.sort(labels), 
                 yerr=[bottom_error, top_error], fmt='o')
    plt.show()
    plt.savefig(PATH_PLOT  + 'error_cac_final_fold' + str(fold) + '.png')
    plt.close()


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


def pre_process_label(mean, std, labels):
    train_score_clip = np.clip(labels.detach().cpu(), a_min=0, a_max=MAX_CAC_VAL)
    train_log_score = np.log(train_score_clip + 0.001)
    return (train_log_score - mean) / std


def mean_std_cac_score_log(loader):
    train_score = torch.cat([labels for (_, labels) in loader]).numpy()
    train_score_clip = np.clip(train_score, a_min=0, a_max=MAX_CAC_VAL)
    train_log_score = np.log(train_score_clip + 0.001)
    return train_log_score.mean(), train_log_score.std()


def norm_labels(mean, std, labels):
    cac_clip = np.clip([labels],a_min=0, a_max=MAX_CAC_VAL)
    log_cac_score = np.log(cac_clip + 0.001)
    return (log_cac_score - mean) / std


def local_copy_str_kfold(dataset):
    data = [dataset[j][0] for j in range(len(dataset))]
    label = [dataset[j][1] for j in range(len(dataset))]
    return data, label


def mean_absolute_error(y_true, y_pred):
    return np.sum(np.abs(np.array(y_true) - np.array(y_pred)))/len(y_true)


def to_class(continuos_values, labels, th):
    classes_labels = [0 if labels[i] <= th else 1 for i in range(labels.size(dim=0))]
    output_labels = [0 if continuos_values[i] <= th else 1 for i in range(continuos_values.size(dim=0))]
    return torch.tensor(output_labels), torch.tensor(classes_labels)


def samples_for_preds(probs):
    samples_for_preds = []
    samples_for_preds.append(((0.5 < probs) & (probs <= 0.6)).sum())
    samples_for_preds.append(((0.6 < probs) & (probs <= 0.7)).sum())
    samples_for_preds.append(((0.7 < probs) & (probs <= 0.8)).sum())
    samples_for_preds.append(((0.8 < probs) & (probs <= 0.9)).sum())
    samples_for_preds.append(((0.9 < probs) & (probs <= 1)).sum())
    return samples_for_preds


def regression_roc_auc_score(y_true, y_pred, num_rounds = 10000):
  """
  Computes Regression-ROC-AUC-score.
  
  Parameters:
  ----------
  y_true: array-like of shape (n_samples,). Binary or continuous target variable.
  y_pred: array-like of shape (n_samples,). Target scores.
  num_rounds: int or string. If integer, number of random pairs of observations. 
              If string, 'exact', all possible pairs of observations will be evaluated.
  
  Returns:
  -------
  rroc: float. Regression-ROC-AUC-score.
  """
      
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)

  num_pairs = 0
  num_same_sign = 0
  
  for i, j in _yield_pairs(y_true, num_rounds):
    diff_true = y_true[i] - y_true[j]
    diff_score = y_pred[i] - y_pred[j]
    if diff_true * diff_score > 0:
      num_same_sign += 1
    elif diff_score == 0:
      num_same_sign += .5
    num_pairs += 1
      
  return num_same_sign / num_pairs


def _yield_pairs(y_true, num_rounds):
  """
  Returns pairs of valid indices. Indices must belong to observations having different values.
  
  Parameters:
  ----------
  y_true: array-like of shape (n_samples,). Binary or continuous target variable.
  num_rounds: int or string. If integer, number of random pairs of observations to return. 
              If string, 'exact', all possible pairs of observations will be returned.
  
  Yields:
  -------
  i, j: tuple of int of shape (2,). Indices referred to a pair of samples.
  
  """  
  if num_rounds == 'exact':
    for i in range(len(y_true)):
      for j in np.where((y_true != y_true[i]) & (np.arange(len(y_true)) > i))[0]:
        yield i, j     
  else:
    for r in range(num_rounds):
      i = np.random.choice(range(len(y_true)))
      j = np.random.choice(np.where(y_true != y_true[i])[0])
      yield i, j
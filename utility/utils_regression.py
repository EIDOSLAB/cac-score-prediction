import torch
import numpy as np
import matplotlib.pyplot as plt

PATH_PLOT = '/home/fiodice/project/plot_training/'
MAX_CAC_VAL = 2000


def cac_prediction_error_bin(labels, preds, mean, std, fold, size_bin):
    labels = np.exp((labels * std) + mean - 0.001).flatten()
    preds = np.exp((preds * std) + mean - 0.001).flatten()
    #labels = np.clip(labels,a_min=0, a_max=2000).flatten()
    #preds = np.clip(preds,a_min=0, a_max=2000).flatten()

    preds_and_labels = list(np.array((preds,labels)).T)
    preds_and_labels.sort(key=lambda pred: pred[0])

    bins, error_bins = [], []
    error_bin = 0
    for index, (prediction, label) in enumerate(preds_and_labels):
        if index == 0:
            bins.append(prediction)

        error_bin += np.abs(label - prediction)

        if index % size_bin == 0 and index != 0:
            bins.append(prediction)
            error_bins.append(error_bin)
            error_bin = 0

    plt.figure(figsize=(18, 10))
    plt.xlabel("Error on bin")
    plt.ylabel("Calcium predicted")
    plt.hist(error_bins, bins=bins)

    plt.show()
    plt.savefig(PATH_PLOT  + 'bin.png')


def cac_prediction_error(labels, preds, mean, std, fold, viz, max_val, log_scale):
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

    preds_and_labels = list(np.array((preds,labels)).T)
    preds_and_labels.sort(key=lambda pred: pred[0])

    top_error, bottom_error = [], []
    for prediction, label in preds_and_labels:
        err = label - prediction
        top_error.append(err)
        bottom_error.append(0)

    if viz:
        preds_sort, labels_sort = zip(*preds_and_labels)
        plt.figure(figsize=(16, 8))
        plt.plot(labels_sort, label='Labels')
        plt.plot(preds_sort, label='Preds')
        plt.legend()
        plt.xlabel("Samples")
        plt.ylabel("Calcium score predicted")
        plt.savefig(PATH_PLOT  + 'error_all_cac_fold' + str(fold) + '.png')
        plt.close()

    plt.figure(figsize=(18, 10))
    plt.xlabel("Samples")
    plt.ylabel("Calcium score predicted")
    plt.grid()
    plt.axhline(y = th, color = 'r', label = "Threshold")

    plt.errorbar(x = np.arange(start=0, stop=len(labels)), 
                 y = np.sort(preds), 
                 yerr=[bottom_error, top_error], fmt='o')
    plt.show()
    plt.savefig(PATH_PLOT  + 'error_cac_fold' + str(fold) + '.png')
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
    #print(train_score.mean(), train_score.std())
    train_score_clip = np.clip(train_score, a_min=0, a_max=MAX_CAC_VAL)
    #print(train_score_clip.mean(), train_score_clip.std())
    train_log_score = np.log(train_score_clip + 0.001)
    #print(train_log_score.mean(), train_log_score.std())
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



if __name__ == '__main__':
    #y = np.array([218, 280, 233, 300, 228, 239]) 
    #x = np.array([102, 90,   90,  20,  0,  0])

    #cac_prediction_error_bin(x,y,1,1,9,2)
    bins = np.array([ 0, 2, 5, 10, 400,1000,2000])
    x = np.array([ 0, 10, 20, 30, 40, 50, 70,90,0,1000,1000,1000,1001])
    w = np.array([ 8, 12, 24, 26, 30, 40, 60])

    plt.figure(figsize=(18, 10))
    plt.xlabel("Error on bin")
    plt.ylabel("Calcium predicted")
    plt.hist(x, bins=bins, weights=w)

    plt.show()
    plt.savefig(PATH_PLOT  + 'bin2.png')

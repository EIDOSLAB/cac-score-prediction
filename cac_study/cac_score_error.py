
import sys
import numpy as np
import sqlite3
from sklearn.utils import resample
import torch
import matplotlib.pyplot as plt


PATH_PLOT = '/home/fiodice/project/plot_training/'



def cac_prediction_error(labels, preds, mean, std, fold):
    #labels = np.exp((labels * std) + mean - 0.001).flatten()
    #preds = np.exp((preds * std) + mean - 0.001).flatten()
    labels = ((labels * std) + mean - 0.001).flatten()
    preds = ((preds * std) + mean - 0.001).flatten()
    labels = np.clip(labels,a_min=0, a_max=2000)

    preds_and_labels = list(np.array((preds,labels)).T)
    preds_and_labels.sort(key=lambda pred: pred[0])

    top_error, bottom_error = [], []

    for prediction, label in preds_and_labels:
        err = label - prediction
        if err < 0:
            top_error.append(0)
            bottom_error.append(err)
        else:
            bottom_error.append(0)
            top_error.append(err)


    print(f'Bottom error {len(bottom_error)} Top error {len(top_error)}')
    plt.figure(figsize=(18, 10))
    plt.axhline(y = 10, color = 'r', label = "Threshold")

    plt.xlabel("Samples")
    plt.ylabel("Calcium score predicted")
    plt.grid()
    plt.errorbar(x = np.arange(start=0, stop=len(labels)), 
                 y = np.sort(preds), 
                 yerr=[bottom_error, top_error], fmt='o')
    plt.show()
    plt.savefig(PATH_PLOT  + 'error_cac_fold' + str(fold) + '.png')
    plt.close()



if __name__ == '__main__':
    #x=np.arange(0,10)
    y=np.array([218, 280, 233, 300, 228, 239])
    x=np.arange(start=0, stop=6)
    #diff = x -y
    up_error = [100,0,0,0,40,50,60]
    bottom_error = [102,90,90,20,0,0,0]

    cac_prediction_error(x,y,1,1,0)

    res = [[prediction, error] for prediction,error in zip(up_error, bottom_error)]
    for prediction, label in res:
        err = prediction - label
        if err < 0:
            bottom_error.append(err)
        else:
            bottom_error.append(0)

    print(res)
    res.sort(key=lambda x: x[0])
    print(res)


    # ridimensioniamo l'immagine
    plt.figure(figsize=(10,10))
    # assegniamo etichette agli assi
    plt.xlabel("Calcium score")
    plt.ylabel("Errore")
    # impostiamo il titolo del grafico
    plt.title("Error")
    # chiediamo di visualizzare la griglia
    plt.grid()
    # disegniamo due linee
    plt.errorbar(x,y,yerr=[bottom_error, up_error],fmt='v')
    plt.show()
    plt.savefig(PATH_PLOT  + 'error_cac2.png')
    plt.close()

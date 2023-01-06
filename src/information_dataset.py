
import numpy as np
import sys
import pandas as pd
import torch
import scipy.stats as stats
import argparse
import sqlite3
import matplotlib.pyplot as plt
import utility.utils as utils
import os
from datetime import date
from tqdm import tqdm
from dataset import CalciumDetection

PATH_PLOT = '/scratch/calcium_score/dataset'
PATH_DATA = '/scratch/dataset/calcium_rx'
#sys.path.insert(0, '/home/fiodice/project/src')

def analize_calcium_distribution(loader):
    scores = []
    cont = 0
    cont2 = 0
    cont3 = 0
    for i,(_, _,labels) in enumerate(loader):
        print(labels)
        if labels.numpy()[0] == 0:
            cont += 1
        elif 1 <= labels.numpy()[0] < 10:
            cont2 += 1
        else:
            cont3 += 1
        scores.append(labels.numpy()[0])
    
    scores = np.clip(scores, a_min=0, a_max=2000)
    
    
    plt.figure()
    #plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
    a = plt.hist(scores, bins=[0, 1, 10, 2001])
    plt.gca().set(title='Frequency Histogram Clip CAC score', xlabel='Calcium score', ylabel='Count')
    plt.savefig(PATH_PLOT + '/cac_frequency.png')
    plt.close()
    
    #a,b = np.histogram(scores,  bins=[0, 1, 10, 2001])
    
    
    
    
    
    
    # scores is the distribution of calcium score. 
    
    # count number of elements for each values of the cac 
    

def main():
    mean, std = [0.5024], [0.2898]


    transform, _ = utils.get_transforms(img_size=1248, crop=1024, mean = mean, std = std)
    cac_samples_data, cac_sample_labels = [], []
    whole_dataset = CalciumDetection(PATH_DATA, transform, mode='classification', require_cac_score=True)
    
    loader = torch.utils.data.DataLoader(whole_dataset, batch_size = 1, shuffle = False, num_workers = 0)
    analize_calcium_distribution(loader)
    
    
    
if __name__ == "__main__":
    main()
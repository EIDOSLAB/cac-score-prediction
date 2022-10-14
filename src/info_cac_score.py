
import numpy as np
import sys
import torch
import scipy.stats as stats
import argparse
import sqlite3
import matplotlib.pyplot as plt
import utility.utils as utils

from datetime import date
from tqdm import tqdm
from dataset import CalciumDetection

PATH_PLOT = '/home/fiodice/project/plot_analyses/'
sys.path.insert(0, '/home/fiodice/project/src')

def save_cac_distribution(loader):
    scores = []

    for (_, labels) in tqdm(loader):
        scores.append(labels.numpy()[0])

    scores = np.clip(scores, a_min=0, a_max=2000)

    plt.figure()
    plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
    plt.hist(scores, bins=int(180/1))
    plt.gca().set(title='Frequency Histogram Clip CAC score', xlabel='Calcium score', ylabel='Count')
    plt.savefig(PATH_PLOT + 'cac_frequency.png')
    plt.close()

    score_clip = np.clip(scores, a_min=0, a_max=2000)
    #print(train_score_clip.mean(), train_score_clip.std())
    log_cac_score = np.log(score_clip + 0.001)

    plt.figure()
    plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
    plt.hist(log_cac_score, bins=int(180/1))
    plt.gca().set(title='Frequency Histogram Log CAC score', xlabel='Calcium score', ylabel='Count')
    plt.savefig(PATH_PLOT + 'cac_log_frequency.png')
    plt.close()
    

path_data = '/home/fiodice/project/dataset/'
path_labels = '/home/fiodice/project/dataset/labels.db'

parser = argparse.ArgumentParser()
parser.add_argument('--distribution', type=bool, default=False, help='save cac distributions')
args = parser.parse_args()

save_distribution = args.distribution
mean, std = [0.5024], [0.2898]

transform, _ = utils.get_transforms(img_size=1248, crop=1024, mean = mean, std = std)
cac_samples_data, cac_sample_labels = [], []

if save_distribution:
    whole_dataset = CalciumDetection(path_data, transform, mode='regression')

    loader = torch.utils.data.DataLoader(whole_dataset,
                               batch_size = 1,
                               shuffle = False,
                               num_workers = 0)
    save_cac_distribution(loader)


conn = sqlite3.connect(path_labels)
conn.row_factory = sqlite3.Row  
cursor = conn.cursor()

labels = [dict(row) for row in cursor.execute('SELECT * FROM patient').fetchall()]

for label in labels:
    cac_sample_labels.append(label['id'])

patients, cac_scores = [], []
ages, rx_date = [], []

sex_cac = {'f' : 0, 'm':0}
samples_without_xr = ['CAC_521', 'CAC_097', 'CAC_017', 'CAC_539', 'CAC_071']

for label in labels:
    if label['id'] not in samples_without_xr:
        patients.append(label['id'])
        cac_scores.append(int(label['cac_score']))   
        sex_cac[label['sex']] += 1
        ages.append(date.today().year - int(label['birth'].split('-')[0]))
        #error on label rx_date yaer = 21
        if(int(label['rx_date'].split('-')[0]) != 21):
            rx_date.append(int(label['rx_date'].split('-')[0]))

ages = np.array(ages)
rx_date = np.array(rx_date)
cac_scores = np.array(cac_scores)


print('Pazienti totali ' + '.'*20 + f' {len(set(patients))}')
print('Pazienti sesso F ' + '.'*20 + f' {sex_cac.get("f")} ')
print('Pazienti sesso M ' + '.'*20 + f' {sex_cac.get("m") }')
print('Pazienti età media ' + '.'*20 + f' {ages.mean():.2f}')
print('Pazienti età minima ' + '.'*20 + f' {ages.min()}')
print('Pazienti età massima ' + '.'*20 + f' {ages.max()}')
print('Pazienti età moda ' + '.'*20 + f' {stats.mode(ages)[0][0]}')
print()

print('Radiografia più recente ' + '.'*20 + f' {rx_date.min()}')
print('Radiografia meno recente ' + '.'*20 + f' {rx_date.max()}')
print()

print('CAC score  Mean:'  + '.'*20 + f'{cac_scores.mean():.2f}')
print('CAC score  Std' + '.'*20 + f' {cac_scores.std():.2f}')
print('CAC score  Median' + '.'*20 + f' {np.ma.median(cac_scores)}')
print('CAC score  Max' + '.'*20 + f' {cac_scores.max()}')
print('CAC score  Min' + '.'*20 + f' {cac_scores.min()}')
print('CAC score  Moda' + '.'*20 + f' {stats.mode(cac_scores)[0][0]:.2f}')
print()

log_cac_score = np.log(np.clip(cac_scores, a_min=0, a_max=2000) + 0.001)

print('Log CAC score  Mean:'  + '.'*20 + f'{log_cac_score.mean():.2f}')
print('Log CAC score  Std' + '.'*20 + f' {log_cac_score.std():.2f}')
print('Log CAC score  Median' + '.'*20 + f' {np.ma.median(log_cac_score):.2f}')
print('Log CAC score  Max' + '.'*20 + f' {log_cac_score.max():.2f}')
print('Log CAC score  Min' + '.'*20 + f' {log_cac_score.min():.2f}')
print('Log CAC score  Moda' + '.'*20 + f' {stats.mode(log_cac_score)[0][0]:.2f}')
print()

norm_log_cac_score = (log_cac_score - log_cac_score.mean())/log_cac_score.std()

print('Norm Log CAC score  Mean:'  + '.'*20 + f'{norm_log_cac_score.mean():.2f}')
print('Norm Log CAC score  Std' + '.'*20 + f' {norm_log_cac_score.std():.2f}')
print('Norm Log CAC score  Median' + '.'*20 + f' {np.ma.median(norm_log_cac_score):.2f}')
print('Norm Log CAC score  Max' + '.'*20 + f' {norm_log_cac_score.max():.2f}')
print('Norm Log CAC score  Min' + '.'*20 + f' {norm_log_cac_score.min():.2f}')
print('Norm Log CAC score  Moda' + '.'*20 + f' {stats.mode(norm_log_cac_score)[0][0]:.2f}')
print()

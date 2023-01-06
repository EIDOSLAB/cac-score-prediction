from email.mime import base
from operator import mod
import torch
import dataset
import sys
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm

from utility import utils
from utility import utils_regression
from models import  cac_detector

#from utility.visualize import *
from utility.config import base_path, seed, TH_cac_score 
import matplotlib.pyplot as plt
#plt.rc('font', size=13) 
#sys.path.insert(0, base_path + '/src')
import seaborn as sns
from sklearn.metrics import confusion_matrix
PATH_PLOT = "/scratch/calcium_score/regression/"


def save_cm(true_labels, best_pred_labels, fold, path_plot):
    cm = confusion_matrix(true_labels, best_pred_labels)
    ax = sns.heatmap(cm, annot=True, fmt="d")
    hm = ax.get_figure()
    hm.savefig(path_plot + 'cm_fold' + str(fold) + '.png')
    hm.clf()
    plt.close(hm)



def pre_process_label(mean, std, labels):
    cac_score_clip = np.clip(labels.detach().cpu(), a_min=0, a_max=max_cac_val)
    log_cac_score = np.log(cac_score_clip + 0.001)
    return (log_cac_score - mean) / std


def inverse_process_label(mean, std, y):
    return np.exp(std*y + mean) - 0.001


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='regressor', help='classifier or regressor')
parser.add_argument('--encoder', type=str, default='densenet121', help='encoder architecture (densenet121 or resnet18 or efficientNet)')
parser.add_argument('--viz', type=bool, default=True, help='save metrics and losses')
parser.add_argument('--modelpath', type=str, 
                                   default='/scratch/calcium_score/models_cac_pt/calcium-detection-seed-42-regr-fold-3_best.pt', 
                                   help='path cac model')

# path best clf model : '/src/models_cac_pt/calcium-detection-clf-seed-42-fold-2_best.pt'
# path best regr model : '/src/models_cac_pt/calcium-detection-seed-42-regr-fold-3_best.pt'

args = parser.parse_args()
visualize_result = args.viz
encoder = args.encoder
mode = args.mode
path_model = args.modelpath


path_data = '/scratch/dataset/calcium_processed'
save_path = '/scratch/calcium_score/regression'
#path_data = base_path + '/dataset/'
#path_labels = base_path + '/dataset/labels.db'
#path_plot = base_path + '/plot_training/'
path_plot = '/scratch/calcium_score/plots'

if encoder == 'densenet121':
    path_encoder =  '/scratch/calcium_score/encoder_pt/dense_final.pt'
elif encoder == 'resnet18':
    path_encoder =  '/scratch/calcium_score/resnet_best.pt'
elif encoder == 'efficientnet-b0':
    path_encoder = '/scratch/calcium_score/eff_best.pt'
else:
    print(f'Unkown encoder_name value: {encoder}')
    exit(1)
    
path_model =  path_model
utils.set_seed(seed)

mean, std = [0.5024], [0.2898]
transform, _ = utils.get_transforms(img_size=1248, crop=1024, mean = mean, std = std)

if mode == 'classifier':
    whole_dataset = dataset.CalciumDetection(path_data, transform, mode='classification', require_cac_score=False)
elif mode == 'regressor':
    whole_dataset = dataset.CalciumDetection(path_data, transform, mode='regression',require_cac_score=True)

dataloader = torch.utils.data.DataLoader(whole_dataset, 
                                        batch_size=1, 
                                        shuffle=False)

model = cac_detector.load_model(path_model, path_encoder, mode)
model.to(device)


mean_cac, std_cac = utils_regression.mean_std_cac_score_log(dataloader)

TH_cac_score_log = (np.log(TH_cac_score + 0.001) - mean_cac) / std_cac

accs, b_accs, auc_scores = np.array([]), np.array([]), np.array([])

true_labels, preds_labels, preds_outputs, true_outputs = [], [], [], []
acc, samples_num = 0., 0
data_xlms = []

for (data, labels, id) in tqdm(dataloader):

    data, labels = data.to(device), labels.to(device)
    print(id)# = id[0]
    with torch.no_grad():
        outputs = model(data)

        if mode == 'classifier':
            _, output_classes = torch.max(outputs, 1)
            labels_class = labels.detach().cpu()
            
            acc += torch.sum(output_classes.detach().cpu() == labels_class.data)

        elif mode == 'regressor':
            #print("sono qua vero?", id,"  ",len(id))
            output_classes, labels_class = utils_regression.to_class(outputs.detach().cpu(), 
                                                          labels.detach().cpu(), 
                                                          TH_cac_score_log)

            
            acc += torch.sum(output_classes == labels_class)
            for i in range(labels.shape[0]):
                c = inverse_process_label(mean_cac, std_cac, outputs[i].detach().cpu().item())
                print(outputs[i].detach().cpu().item(),"  ",c,"  ", labels[i].detach().cpu())
                data_xlms.append([id[i], labels[i].detach().cpu().item(),int(c),labels_class[i].detach().cpu().item(), output_classes[i].detach().cpu().item()])
        
            
        preds_labels.append(output_classes.cpu())
        true_labels.append(labels_class.cpu())

        samples_num += len(labels)

df = pd.DataFrame(data_xlms, columns=['id', 'real_calcium_score', 'predicted_calcium_score',"real_class","predicted_class"])
df.to_excel(save_path + "/results.xlsx")
print(f'Model test accuracy: {acc/samples_num:.4f}')
path_plot = os.path.join(path_plot,"final")
save_cm(torch.cat(true_labels).numpy(), torch.cat(preds_labels).numpy(), 0, path_plot)
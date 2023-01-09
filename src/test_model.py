from email.mime import base
from operator import mod
import torch
import dataset
import sys
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
import os
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


def save_cm(true_labels, best_pred_labels, path_plot):
    pth = os.path.join( path_plot,'cm_fold_external.png' )
    cm = confusion_matrix(true_labels, best_pred_labels)
    ax = sns.heatmap(cm, annot=True, fmt="d")
    hm = ax.get_figure()
    hm.savefig(pth)
    hm.clf()
    plt.close(hm)



def pre_process_label(mean, std, labels):
    cac_score_clip = np.clip(labels.detach().cpu(), a_min=0, a_max=max_cac_val)
    log_cac_score = np.log(cac_score_clip + 0.001)
    return (log_cac_score - mean) / std


def inverse_process_label(mean, std, y):
    return np.exp(std*y + mean) - 0.001



def get_args(): 

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='classification', help='classifier or regressor')
    parser.add_argument('--encoder', type=str, default='densenet121', help='encoder architecture (densenet121 or resnet18 or efficientNet)')
    parser.add_argument('--viz', type=bool, default=True, help='save metrics and losses')
    parser.add_argument('--path_base', type=str, default='/scratch/calcium_score/', help='path cac model') 
    parser.add_argument('--path_data', type=str, default='/scratch/dataset/calcium_processed/', help='path cac model') 
    parser.add_argument('--path_encoder', type=str, default= '/scratch/calcium_score/encoder_pt/' , help='path cac model')
    parser.add_argument('--internal_data', type = bool, default=False, help = "internal or external data of base hospitals")
    parser.add_argument('--require_cac_score', type = bool, default=True, help = "cac score output")
    
    args = parser.parse_args()   
    return args



def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    # path best clf model : '/src/models_cac_pt/calcium-detection-clf-seed-42-fold-2_best.pt'
    # path best regr model : '/src/models_cac_pt/calcium-detection-seed-42-regr-fold-3_best.pt'


    visualize_result = args.viz
    encoder = args.encoder
    mode = args.mode
    path_base = args.path_base

    # define models
    if mode == "classification":
        path_models = os.path.join(path_base,"classification","models",'Best-calcium-detection-clf-seed-42-fold-3.pt')
    else:
        path_models = os.path.join(path_base,"regression","models", 'calcium-detection-seed-42-regr-fold-3_best.pt')
        

    path_data = args.path_data
    path_encoder = args.path_encoder
    internal_data = args.internal_data
    require_cac_score = args.require_cac_score



    path_save_results = os.path.join(path_base,mode,"prediction")
    path_plot = os.path.join(path_base,mode,"plots")

    if encoder == 'densenet121':
        path_encoder =  os.path.join(path_encoder,"dense_final.pt")
    elif encoder == 'resnet18':
        path_encoder = os.path.join(path_encoder,"resnet_best.pt") 
    elif encoder == 'efficientnet-b0':
        path_encoder = os.path.join(path_encoder,"eff_best.pt") 
    else:
        print(f'Unkown encoder_name value: {encoder}')
        exit(1)




    utils.set_seed(seed)

    mean, std = [0.5024], [0.2898]
    transform, _ = utils.get_transforms(img_size=1248, crop=1024, mean = mean, std = std)


    whole_dataset = dataset.CalciumDetection(path_data, 
                                                transform, 
                                                mode=args.mode, 
                                                require_cac_score=require_cac_score, 
                                                internal_data = internal_data)



    dataloader = torch.utils.data.DataLoader(whole_dataset, 
                                            batch_size=4, 
                                            shuffle=False)

    model = cac_detector.load_model(path_models, path_encoder, mode, encoder)
    model.to(device)


    print("calcolo std mean")
    if mode == "regression":
        mean_cac, std_cac = utils_regression.mean_std_cac_score_log(dataloader)
        TH_cac_score_log = (np.log(TH_cac_score + 0.001) - mean_cac) / std_cac

    accs, b_accs, auc_scores = np.array([]), np.array([]), np.array([])

    true_labels, preds_labels, preds_outputs, true_outputs = [], [], [], []
    acc, samples_num = 0., 0
    data_xlms = []


    print("inizio test!")
    for (data, labels,  cac_score, id) in tqdm(dataloader):
               
        # labels = 0/1 cac_score = cac_score
        data, labels, cac_score  = data.to(device), labels.to(device), cac_score.to(device) # for regression label is useless
        with torch.no_grad():
            outputs = model(data)
            if mode == 'classification':
                _, output_classes = torch.max(outputs, 1)
                labels_class = labels.detach().cpu()           
                acc += torch.sum(output_classes.detach().cpu() == labels_class.data)           
                for i in range(cac_score.shape[0]):
                    data_xlms.append([id[i], cac_score[i].detach().cpu().item(),"_",labels_class[i].detach().cpu().item(), output_classes[i].detach().cpu().item()])
        
            elif mode == 'regresssion':           
                output_classes, labels_class = utils_regression.to_class(outputs.detach().cpu(), 
                                                            cac_score.detach().cpu(), 
                                                            TH_cac_score_log)

                
                acc += torch.sum(output_classes == labels_class)
                for i in range(cac_score.shape[0]):
                    c = inverse_process_label(mean_cac, std_cac, outputs[i].detach().cpu().item())
                    data_xlms.append([id[i], cac_score[i].detach().cpu().item(),int(c),labels_class[i].detach().cpu().item(), output_classes[i].detach().cpu().item()])
            
            else:
                print(f'Unkown mode value: {mode}')
                exit(1)
                
            preds_labels.append(output_classes.cpu())
            true_labels.append(labels_class.cpu())

            samples_num += len(labels)

    df = pd.DataFrame(data_xlms, columns=['id', 'real_calcium_score', 'predicted_calcium_score',"real_class","predicted_class"])
    df.to_excel(os.path.join(path_save_results,"results.xlsx"))
    print(f'Model test accuracy: {acc/samples_num:.4f}')
    save_cm(torch.cat(true_labels).numpy(), torch.cat(preds_labels).numpy(), path_plot)
    
    
    


if __name__ == '__main__':
    

    print(f'torch version: {torch.__version__}')
    print(f'cuda version: {torch.version.cuda}')

    args = get_args()
    main(args)
from email.mime import base
from operator import mod
import pandas as pd
import torch
import dataset
import copy
import itertools
import sys
import numpy as np
import argparse
import os
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from utility import utils
from models import  cac_detector

from utility.visualize import *
from utility.config import base_path, seed



def run(model, dataloader, criterion, optimizer,device, scheduler=None, phase='train'):
    epoch_loss, epoch_acc, samples_num = 0., 0., 0.
    true_labels, pred_labels, outputs_labels = [], [], []
    all_cac_scores = []
    probabilities = None
  
    for (data, labels, cac_scores,_) in tqdm(dataloader):
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels).to(device)

        true_labels.append(labels.detach().cpu())
        pred_labels.append(preds.detach().cpu())
        outputs_labels.append(outputs.detach().cpu())
        all_cac_scores.append(cac_scores)

        if phase == 'train':
            loss.backward()
            optimizer.step()
        
        epoch_loss += loss.detach().cpu().item()
        epoch_acc += torch.sum(preds == labels.data)
        samples_num += len(labels)

    if scheduler is not None and phase == 'train':
        scheduler.step()
    
    if phase == 'test':
        probabilities = torch.nn.functional.softmax(torch.cat(outputs_labels), dim=1)
        
    return epoch_loss / len(dataloader), epoch_acc / samples_num, torch.cat(true_labels).numpy(), torch.cat(pred_labels).numpy(), probabilities, torch.cat(all_cac_scores).numpy() 





def get_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50, help='num. of epochs (default 50)')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate default (3e-4)')
    parser.add_argument('--arch', type=str, default='densenet121', help='encoder architecture (densenet121 or resnet18 or efficientNet)')
    parser.add_argument('--viz', type=bool, default=True, help='save metrics and losses')
    parser.add_argument('--save', type=bool, default=True, help='save model')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay value (default 1e-4)')
    parser.add_argument('--batchsize', type=float, default=4, help='batch size value (default 4)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum value (default 0.9)')
    parser.add_argument('--kfold', type=int, default=5, help='folds for cross-validation (default 5)')
    parser.add_argument('--seed',type=int, default=42, help='seed value (default 42)')
    parser.add_argument('--base_path',type= str, default='/scratch/calcium_score', help='base path ')
    parser.add_argument('--path_data',type=str, default = '/scratch/dataset/calcium_rx/', help = 'path for data')

    
    args = parser.parse_args()   
    return args










def main(args):
    utils.set_seed(args.seed)
    lr = args.lr
    encoder_name = args.arch
    epochs = args.epochs
    visualize_result = args.viz
    save_model = args.save
    weight_decay = args.wd
    momentum = args.momentum
    k_folds = args.kfold
    batchsize = args.batchsize
    
    mean, std = [0.5024], [0.2898]

    
    
    base_path = args.base_path
    path_data = args.path_data
    path_plot = os.path.join(base_path,'classification', 'plots')
    models_path= os.path.join(base_path,'classification', 'models')
    encoder_path = os.path.join(base_path,'encoder_pt')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    
    if encoder_name == 'densenet121':
        path_model = os.path.join(encoder_path,'dense_final.pt')
    elif encoder_name == 'resnet18':
        path_model = os.path.join(encoder_path,'resnet_best.pt')
    elif encoder_name == 'efficientnet-b0':
        path_model = os.path.join(encoder_path,'eff_best.pt')
    else:
        print(f'Unkown encoder_name value: {encoder_name}')
        exit(1)

    transform, _ = utils.get_transforms(img_size=1248, crop=1024, mean = mean, std = std)

    whole_dataset = dataset.CalciumDetection(path_data, transform, mode='classification', require_cac_score=True, internal_data = True)
    # From torch.utils.data.Dataset to list, this increse speed of each epochs
    whole_dataset = utils.local_copy(whole_dataset, require_cac_score=True)

    kfold = KFold(n_splits=k_folds, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()

    accs, b_accs, auc_scores = np.array([]), np.array([]), np.array([])

    for fold, (train_ids, test_ids) in enumerate(kfold.split(whole_dataset)):
        print('\n','='*20, f'Fold: {fold}','='*20,'\n')

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        train_loader = torch.utils.data.DataLoader(
                        whole_dataset, 
                        batch_size=batchsize, sampler=train_subsampler)

        test_loader = torch.utils.data.DataLoader(
                        whole_dataset,
                        batch_size=batchsize, sampler=test_subsampler)

        #if visualize_result:
        #    viz_distr_data_binary(train_loader, 'train', fold)
        #    viz_distr_data_binary(test_loader, 'test', fold)
        
        model = cac_detector.CalciumDetector(encoder = encoder_name, path_encoder = path_model, mode='classification').to(device)
        encoder_last_layer = cac_detector.unfreeze_lastlayer_encoder(model, encoder_name)

        best_model = None
        best_model_probs = []

        test_acc, test_loss = 0., 0.
        best_test_acc, best_test_bacc = 0., 0.
        roc_auc = 0

        best_model_pred_labels, best_model_cac_scores  = [], []
        true_labels, pred_labels = [], []
        train_losses, test_losses = [], []

        params = [model.fc.parameters(), encoder_last_layer.parameters()]
        optimizer = torch.optim.AdamW(itertools.chain(*params), betas=(0.9,0.999),eps=1e-08, lr=lr, weight_decay=weight_decay)
        scheduler = MultiStepLR(optimizer, milestones=[20, 35], gamma=0.1)

        for epoch in range(1, epochs+1):
            print('\n','='*20, f'Epoch: {epoch}','='*20,'\n')

            train_loss, train_acc, _, _, _, _ = run(model, train_loader, criterion, optimizer, device,scheduler=scheduler)
            test_loss, test_acc, true_labels, pred_labels, probs, cac_scores = run(model, test_loader, criterion, optimizer,device,scheduler=scheduler, phase='test')
            test_bacc = balanced_accuracy_score(true_labels, pred_labels)

            print(f'\nTrain loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}')
            print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}\n')
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            if best_model is None or (test_bacc > best_test_bacc):
                best_model = copy.deepcopy(model)
                best_model_cac_scores = cac_scores
                best_test_bacc = balanced_accuracy_score(true_labels, pred_labels)
                best_true_label = true_labels
                best_test_acc = test_acc 
                best_model_pred_labels = pred_labels 
                best_model_probs = probs

                print(f'Model UPDATE Acc: {accuracy_score(true_labels, pred_labels):.4f} B-Acc : {balanced_accuracy_score(true_labels, pred_labels):.4f}')
                save_cm(true_labels, best_model_pred_labels, fold, path_plot)
                roc_auc = save_roc_curve(true_labels, best_model_probs[:, 1], fold, path_plot)
                torch.save({'model': best_model.state_dict()}, os.path.join(models_path,f'Best-calcium-detection-clf-seed-{seed}-fold-{fold}.pt'))

            if save_model:
                torch.save({'model': best_model.state_dict()}, os.path.join(models_path,f'calcium-detection-clf-seed-{seed}-fold-{fold}.pt'))
    
        max_probabilities = [max(np.array(probs)) for probs in best_model_probs] 

        if visualize_result: 
            save_losses(train_losses, test_losses, best_test_acc, fold, path_plot)
            #viz_probs_prediction(max_probabilities, best_model_pred_labels, best_true_label, fold, path_plot)
            #viz_samples_missclf(max_probabilities, best_model_cac_scores, best_model_pred_labels, best_true_label, fold, path_plot)
        
        b_accs = np.append(b_accs, best_test_bacc)
        accs = np.append(accs, best_test_acc.cpu())
        auc_scores = np.append(auc_scores, roc_auc)

       

    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    results_xlsx = []
    for fold, value in enumerate(accs):
        print(f'Fold {fold}: Acc {accs[fold]:.4f} BA {b_accs[fold]:.4f} AUC {auc_scores[fold]:.4f}\n')
        results_xlsx.append([fold,accs[fold], b_accs[fold],auc_scores[fold]])

    print(f'Average  AC: {accs.mean():.4f} BA: {b_accs.mean():.4f} AUC: {auc_scores.mean():.4f}\n')
    results_xlsx.append([10,accs.mean(), b_accs.mean(),auc_scores.mean()])
    df = pd.DataFrame(results_xlsx, columns=['fold', 'accuracy', 'balanced_accuracy',"auc_scores"])
    
    df.to_excel(os.path.join(path_plot,"folded_results.xlsx"))
    
    
    


if __name__ == '__main__':
    

    print(f'torch version: {torch.__version__}')
    print(f'cuda version: {torch.version.cuda}')

    args = get_args()
    main(args)
from email.mime import base
from operator import mod
import torch
import dataset
import copy
import itertools
import sys
import numpy as np
import argparse

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from utility import utils_regression
from utility import utils
from utility import metrics
from models import cac_detector

from utility.visualize import *
from utility.config import base_path, seed, TH_cac_score



def run(model, dataloader, criterion, optimizer, mean, std, scheduler=None, phase='train'):
    epoch_loss, epoch_acc, samples_num = 0., 0., 0.
    true_labels, pred_labels = [], []
    all_labels, all_outputs = [], []
    run_abs = 0.,

    for (data, _, labels,_) in tqdm(dataloader):
        labels = utils_regression.pre_process_label(mean, std, labels)

        optimizer.zero_grad()
        with torch.set_grad_enabled(phase == 'train'):
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)

            TH_cac_score_log = (np.log(TH_cac_score + 0.001) - mean) / std
            output_classes, labels_classes = utils_regression.to_class(outputs.detach().cpu(), 
                                                                       labels.detach().cpu(), 
                                                                       TH_cac_score_log)

            loss = criterion(outputs.float(), labels.unsqueeze(dim=1).float()).to(device)

        true_labels.append(labels_classes)
        pred_labels.append(output_classes)
        all_labels.append(labels.detach().cpu())
        all_outputs.append(outputs.detach().cpu())
        
        if phase == 'train':
            loss.backward()
            optimizer.step()
        
        epoch_loss += loss.detach().cpu().item()
        epoch_acc += torch.sum(labels_classes == output_classes)
        samples_num += len(labels) 
        run_abs += metrics.mean_absolute_error(labels.detach().cpu(), outputs.detach().cpu())
        
    if scheduler is not None and phase == 'train':
        scheduler.step()

    return epoch_loss / len(dataloader), epoch_acc / samples_num, torch.cat(true_labels).numpy(), torch.cat(pred_labels).numpy(), torch.cat(all_labels).numpy() , torch.cat(all_outputs).numpy(), run_abs / samples_num 



def get_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=90, help='num. of epochs (default 90)')
    parser.add_argument('--seed',type= int, default = 42, help = 'seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate default (0.001)')
    parser.add_argument('--arch', type=str, default='densenet121', help='encoder architecture (densenet121 or resnet18 or efficientnet-b0) (default densenet121)')
    parser.add_argument('--viz', type=bool, default='True', help='save metrics and losses  (default True)')
    parser.add_argument('--save', type=bool, default='False', help='save model (default True)')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay value (default 1e-4)')
    parser.add_argument('--batchsize', type=float, default=4, help='batch size value (default 4)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum value (default 0.9)')
    parser.add_argument('--kfolds', type=int, default=5, help='folds for cross-validation (default 5)')
    parser.add_argument('--loss', type=str, default='MAE', help='loss function (MSE or MAE) (default MAE)')
    parser.add_argument('--base_path',type= str, default='/scratch/calcium_score', help='base path ')
    parser.add_argument('--path_data',type=str, default = '/scratch/dataset/calcium_rx/', help = 'path for data')
    parser.add_argument('--mean',type= float, default = 0.5024, help = 'mean')
    parser.add_argument('--std',type= float, default = 0.2898, help = 'std')
    
    args = parser.parse_args()   
    return args


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed = args.seed
    args = parser.parse_args()
    utils.set_seed(seed)

    lr = args.lr
    encoder_name = args.arch
    epochs = args.epochs
    visualize_result = args.viz
    save_model = args.save
    weight_decay = args.wd
    momentum = args.momentum
    k_folds = args.kfolds
    batchsize = args.batchsize
    loss = args.loss
    
    path_data = args.path_data
    base_path = args.base_path
    
    path_plot = os.path.join(base_path,"regression","plots")
    encoder_path = os.path.join(base_path,'encoder_pt')
    models_path= os.path.join(base_path,"regression", 'models')
    
    #unfreeze_last_layer_encoder = args.layer_enc_freeze
    # From CheXpert
    mean, std = [args.mean], [args.std]

    if encoder_name == 'densenet121':
        path_model = os.path.join(encoder_path,'dense_final.pt')
    elif encoder_name == 'resnet18':
        path_model = os.path.join(encoder_path,'resnet_best.pt')
    elif encoder_name == 'efficientnet-b0':
        path_model = os.path.join(encoder_path,'eff_best.pt')
    else:
        print(f'Unkown encoder_name value: {encoder_name}')
        exit(1)


    if loss == 'MAE':
        criterion = torch.nn.L1Loss()
    elif loss == 'MSE':
        criterion = torch.nn.MSELoss()
    else:
        print(f'Unkown loss value: {loss}')
        exit(1)

    transform, _ = utils.get_transforms(img_size=1248, crop=1024, mean = mean, std = std)

    whole_dataset = dataset.CalciumDetection(path_data, transform, mode='regression')
    # From torch.utils.data.Dataset to list, this increse speed of each epochs
    whole_dataset = utils.local_copy(whole_dataset, require_cac_score=True)
    datas, labels = utils_regression.local_copy_str_kfold(whole_dataset)

    skf = StratifiedKFold(n_splits= k_folds)

    accs, b_accs, auc_scores = np.array([]), np.array([]), np.array([])
    print("INIZIO TRAINING")
    for fold, (train_ids, test_ids) in enumerate(skf.split(datas, labels)):
        print('\n','='*20, f'Fold: {fold}','='*20,'\n')

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        train_loader = torch.utils.data.DataLoader(
                        whole_dataset, 
                        batch_size=batchsize, sampler=train_subsampler)

        mean_cac, std_cac = utils_regression.mean_std_cac_score_log(train_loader)

        test_loader = torch.utils.data.DataLoader(
                        whole_dataset,
                        batch_size=batchsize, sampler=test_subsampler)

        if visualize_result:
            viz_distr_data(train_loader, fold,path_plot, 'train', path_plot)
            viz_distr_data(test_loader, fold,path_plot, 'test',path_plot)
        
        model = cac_detector.CalciumDetector(encoder = encoder_name, path_encoder = path_model, mode='regressor').to(device)
        last_layer = cac_detector.unfreeze_lastlayer_encoder(model, encoder_name)
        params = [model.fc.parameters(), last_layer.parameters()]

        best_model = None
        test_acc, test_loss = 0., 0.
        best_test_acc, best_test_bacc = 0., 0.
        best_model_pred_labels, best_model_true_label = [], []

        train_losses, test_losses = [], []
        train_accs, test_accs = [],[]
        trains_abs, tests_abs = [], []
        best_model_outputs = []
        
        optimizer = torch.optim.SGD(itertools.chain(*params),  lr=lr, weight_decay=weight_decay, momentum=momentum)
        scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.1)
        #print(f'Pytorch trainable param {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

        for epoch in range(1, epochs+1):
            print('\n','='*20, f'Epoch: {epoch}','='*20,'\n')

            train_loss, train_acc, _, _, _, _, train_abs = run(model, train_loader, criterion, optimizer, mean=mean_cac, std=std_cac, scheduler=scheduler)
            test_loss, test_acc, class_labels, class_preds, labels, outputs, test_abs = run(model, test_loader, criterion, optimizer, mean=mean_cac, std=std_cac, scheduler=scheduler, phase='test')

            print(f'\nTrain loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}')
            print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}\n')

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            trains_abs.append(train_abs)
            tests_abs.append(test_abs)

            if best_model is None or (test_acc > best_test_acc):
                best_model = copy.deepcopy(model)
                best_test_bacc = balanced_accuracy_score(class_labels, class_preds)
                best_test_acc = test_acc 
                best_model_true_label = labels
                best_model_pred_labels = class_preds 
                best_model_outputs = outputs

                print(f'Model UPDATE Acc: {best_test_acc:.4f} B-Acc : {best_test_bacc:.4f}')
                save_cm(class_labels, best_model_pred_labels, fold, path_plot)
                viz_cac_error(best_model_true_label, best_model_outputs, mean_cac, std_cac, fold, path_plot,log_scale=True)
                viz_cac_error_bins(best_model_true_label, best_model_outputs, mean_cac, std_cac, path_plot,fold)
                torch.save({'model': best_model.state_dict()}, os.path.join(models_path,f'Best-NEW-calcium-detection-seed-{seed}-regr-fold-{fold}.pt'))


        save_metric(train_accs, test_accs, 'accs', fold, path_plot)
        save_metric(trains_abs, tests_abs, 'abs', fold, path_plot)   


        print('--------------------------------')
        
        auc_score = metrics.regression_roc_auc_score(class_labels, class_preds)

        b_accs = np.append(b_accs, best_test_bacc)
        accs = np.append(accs, best_test_acc.cpu())
        auc_scores = np.append(auc_scores, auc_score)

        #save_losses(train_losses, test_losses, best_test_acc, fold)

    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')

    results_xlsx = []
    for fold, value in enumerate(accs):
        print(f'Fold {fold}: Acc {accs[fold]:.4f} BA {b_accs[fold]:.4f} AUC {auc_scores[fold]:.4f}\n')
        results_xlsx.append([fold,accs[fold], b_accs[fold],auc_scores[fold]])
        
    results_xlsx.append([10,accs.mean(), b_accs.mean(),auc_scores.mean()])
    df = pd.DataFrame(results_xlsx, columns=['fold', 'accuracy', 'balanced_accuracy',"auc_scores"])
    df.to_excel(os.path.join(path_plot,"folded_results.xlsx")) 
    print(f'Average  AC: {accs.mean():.4f} BA: {b_accs.mean():.4f} AUC: {auc_scores.mean():.4f}\n')
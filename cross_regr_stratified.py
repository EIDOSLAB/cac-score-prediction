import torch
import dataset
import copy
import itertools
import sys
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from collections import Counter
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from utility import utils_regression
from utility import utils
from models import utils_model

sys.path.insert(0, '/home/fiodice/project/src')

PATH_PLOT = '/home/fiodice/project/plot_training/'
THRESHOLD_CAC_SCORE = 10
VISUALIZE_FOLD = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run(model, dataloader, criterion, optimizer, mean, std, scheduler=None, phase='train'):
    epoch_loss, epoch_acc, samples_num = 0., 0., 0.
    true_labels, pred_labels = [], []
    all_labels, all_outputs = [], []
    run_abs = 0.,

    for (data, labels) in tqdm(dataloader):
        labels = utils_regression.pre_process_label(mean, std, labels)

        optimizer.zero_grad()
        with torch.set_grad_enabled(phase == 'train'):
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)

            th = (np.log(THRESHOLD_CAC_SCORE + 0.001) - mean) / std
            output_classes, labels_classes = utils_regression.to_class(outputs.detach().cpu(), labels.detach().cpu(), th)

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
        run_abs += utils_regression.mean_absolute_error(labels.detach().cpu(), outputs.detach().cpu())

    if scheduler is not None and phase == 'train':
        scheduler.step()

    return epoch_loss / len(dataloader), epoch_acc / samples_num, torch.cat(true_labels).numpy(), torch.cat(pred_labels).numpy(), torch.cat(all_labels).numpy() , torch.cat(all_outputs).numpy(), run_abs / samples_num 
   
  
if __name__ == '__main__':
    path_data = '/home/fiodice/project/dataset/'
    path_labels = '/home/fiodice/project/labels/labels_new.db'
    path_model = '/home/fiodice/project/src/models_pt/pretrained_model/dense_final.pt'

    accs, b_accs = [], []

    seed = 42
    k_folds = 5
    epochs = 40
    batchsize = 4
    mean, std = [0.5024], [0.2898]

    utils.set_seed(seed)
    transform, _ = utils.get_transforms(img_size=1248, crop=1024, mean = mean, std = std)

    whole_dataset = dataset.CalciumDetectionRegression(path_data, path_labels, transform)
    whole_dataset = utils.local_copy(whole_dataset)
    datas, labels = utils_regression.local_copy_str_kfold(whole_dataset)

    skf = StratifiedKFold(n_splits= k_folds)
    criterion = torch.nn.MSELoss()

    print('='*30)
    for fold, (train_ids, test_ids) in enumerate(skf.split(datas, labels)):

        print(f'FOLD {fold}')
        print('='*30)
    
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        train_loader = torch.utils.data.DataLoader(
                        whole_dataset, 
                        batch_size=batchsize, sampler=train_subsampler)

        mean_cac, std_cac = utils_regression.mean_std_cac_score_log(train_loader)

        test_loader = torch.utils.data.DataLoader(
                        whole_dataset,
                        batch_size=batchsize, sampler=test_subsampler)

        if VISUALIZE_FOLD:
            utils_regression.viz_distr_data(train_loader, fold, 'train')
            utils_regression.viz_distr_data(test_loader, fold, 'test')
        
        model = utils_model.densenet_regressor(path_model)

        model.to(device)
        model_last_layer = utils_model.unfreeze_param_lastlayer_dense_regr(model)

        best_model = None
        best_test_acc, best_test_bacc = 0., 0.
        best_pred_labels = []
        true_labels = []
        pred_labels = []
        test_acc = 0.
        test_loss = 0.
        
        lr = 0.001
        weight_decay = 0.0001
        momentum = 0.9
        
        params = [model.fc.parameters(), model_last_layer.parameters()]
        optimizer = torch.optim.SGD(itertools.chain(*params),  lr=lr, weight_decay=weight_decay, momentum=momentum)
        scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.1)

        train_losses, test_losses = [], []
        train_accs, test_accs = [],[]
        trains_abs, tests_abs = [], []
        f_outputs = []

        print(f'Pytorch trainable param {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

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
                best_pred_labels = class_preds 
                f_outputs = outputs

                print(f'Model UPDATE Acc: {accuracy_score(class_labels, class_preds):.4f} B-Acc : {balanced_accuracy_score(class_labels, class_preds):.4f}')
                print(f'Labels {Counter(class_labels)} Output {Counter(best_pred_labels)}')
                utils.save_cm_fold(class_labels, best_pred_labels, fold, PATH_PLOT)
        
        utils_regression.cac_prediction_error(labels, f_outputs, mean_cac, std_cac, fold, True, 300, True)
        #utils_regression.cac_prediction_error_log(labels, f_outputs, mean_cac, std_cac, fold, True, 300)

        torch.save({'model': best_model.state_dict()}, f'calcium-detection-sdg-seed-{seed}-regr-fold-{fold}.pt')

        utils.save_metric_fold(train_accs, test_accs, 'accs', fold, PATH_PLOT)
        utils.save_metric_fold(trains_abs, tests_abs, 'abs', fold, PATH_PLOT)
        
        print('Accuracy for fold %d: %d %%' % (fold, 100.0 * best_test_acc))
        print('B-Accuracy for fold %d: %d %%' % (fold, 100.0 * best_test_bacc))

        print('--------------------------------')
        
        b_accs.append(100.0 * best_test_bacc)
        accs.append(100.0 * best_test_acc)
        utils.save_losses_fold(train_losses, test_losses, best_test_acc, fold, PATH_PLOT)

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    
    b_accs = np.array(b_accs)
    accs = np.array(accs)

    for fold, value in enumerate(accs):
        print(f'Fold {fold}: {value} %')
    print(f'ACC Average: {accs.mean()} % STD : {accs.std()}\n')

    for fold, value in enumerate(b_accs):
        print(f'Fold {fold}: {value} %')
    print(f'B-ACC Average: {b_accs.mean()} % STD : {b_accs.std()}\n')

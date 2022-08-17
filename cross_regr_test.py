import torch
import dataset
import sys
import numpy as np

from sklearn.model_selection import KFold
from tqdm import tqdm

from utility import utils_regression
from utility import utils
from models import utils_model

sys.path.insert(0, '/home/fiodice/project/src')

PATH_PLOT = '/home/fiodice/project/plot_training/'
THRESHOLD_CAC_SCORE = 10
VISUALIZE_FOLD = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    path_data = '/home/fiodice/project/dataset/'
    path_labels = '/home/fiodice/project/labels/labels_new.db'
    path_model = '/home/fiodice/project/src/models_pt/final/calcium-detection-sdg-seed-42-regr-fold-1.pt'

    accs, b_accs = [], []

    seed = 42
    k_folds = 5
    batchsize = 4
    mean, std = [0.5024], [0.2898]

    utils.set_seed(seed)
    transform, _ = utils.get_transforms(img_size=1248, crop=1024, mean = mean, std = std)

    whole_dataset = dataset.CalciumDetectionRegression(path_data, path_labels, transform)

    kfold = KFold(n_splits=k_folds, shuffle=True)
    criterion = torch.nn.MSELoss()

    print('='*30)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(whole_dataset)):

        print(f'FOLD {fold}')
        print('='*30)
    
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        train_loader = torch.utils.data.DataLoader(
                        whole_dataset, 
                        batch_size=batchsize, sampler=train_subsampler)

        mean_cac, std_cac = 1,1

        test_loader = torch.utils.data.DataLoader(
                        whole_dataset,
                        batch_size=batchsize, sampler=test_subsampler)


        model = utils_model.test_densenet_regressor(path_model)

        all_labels, all_outputs = [], []
        true_labels, pred_labels  = [], []

        for (data, labels) in tqdm(test_loader):
            labels = utils_regression.pre_process_label(mean_cac, std_cac, labels)
            
            with torch.set_grad_enabled(False):
                outputs = model(data)

                th = (np.log(THRESHOLD_CAC_SCORE + 0.001) - mean) / std

                all_labels.append(labels.detach().cpu())
                all_outputs.append(outputs.detach().cpu())
        
        utils_regression.cac_prediction_error2(torch.cat(all_labels).numpy(), torch.cat(all_outputs).numpy(), mean_cac, std_cac, 0, False, 300, False)
        utils_regression.cac_prediction_error(torch.cat(all_labels).numpy(), torch.cat(all_outputs).numpy(), mean_cac, std_cac, 0, False, 300, False)


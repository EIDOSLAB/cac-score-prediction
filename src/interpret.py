import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from utility import utils_regression
import torchvision
from torchvision import models
from torchvision import transforms

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
import os
from model import load_densenet_mlp, HierarchicalResidual
from utils import dicom_img



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
    parser.add_argument('--path_train_data',type=str, default = '/scratch/dataset/calcium_rx/', help = 'path for training data')
    parser.add_argument('--path_test_data',type=str, default = '/scratch/dataset/calcium_processed/', help = 'path for test data')
    parser.add_argument('--mean',type= float, default = 0.5024, help = 'mean')
    parser.add_argument('--std',type= float, default = 0.2898, help = 'std'),
    parser.add_argument('--TH_cac_score',type= float, default = 1.0, help = 'std')
    args = parser.parse_args()   
    return args


def load_cac_detect(path_model):
    model = HierarchicalResidual(encoder='densenet121')
    del model.fc1
    del model.fc2
    
    model.fc =  torch.nn.Sequential(
            torch.nn.Linear(1024, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2))
            
    dict_model = torch.load(path_model)["model"]
    model.load_state_dict(dict_model)

    return model


def to_class(continuos_values,  th):
    output_labels = [0 if continuos_values[i] <= th else 1 for i in range(continuos_values.size(dim=0))]
    return torch.tensor(output_labels)



def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path_model = args.path_model
    model_name = args.model_name
    TH_cac_score = args.TH_cac_score

    path = os.path.join(path_model, model_name)
    model = load_cac_detect(path)
    model.to(device)

    mean, std = [0.5024], [0.2898]

    transform = transforms.Compose([ transforms.Resize((1248,1248)),
                                    transforms.CenterCrop(1024),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std),
    ])

    path_img = args.path_img


    img, _ = dicom_img(path_img)

    transformed_img = transform(img)

    input = transformed_img.unsqueeze(0)
    input = input.to(device)

    output = model(input)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()
    print('Predicted:', pred_label_idx, '(', prediction_score.squeeze().item(), ')')


    integrated_gradients = IntegratedGradients(model)
    attributions_ig = integrated_gradients.attribute(input, target=pred_label_idx, n_steps=3)   


    default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)

    input_attr = np.transpose(attributions_ig.squeeze(0).cpu().detach().numpy(), (1,2,0))
    input_img = np.transpose(input.squeeze(0).cpu().detach().numpy(), (1,2,0))

    _ = viz.visualize_image_attr(input_attr,
                             input_img,
                             method='heat_map',
                             cmap=default_cmap,
                             show_colorbar=True,
                             sign='positive',
                             outlier_perc=1)

    save_path = args.save_path
    plt.savefig(save_path)


    noise_tunnel = NoiseTunnel(integrated_gradients)

    attributions_ig_nt = noise_tunnel.attribute(input, nt_samples=10, nt_type='smoothgrad_sq', target=pred_label_idx)
    _ = viz.visualize_image_attr_multiple(np.transpose(attributions_ig_nt.squeeze(0).cpu().detach().numpy(), (1,2,0)),
                                        np.transpose(transformed_img.squeeze(0).cpu().detach().numpy(), (1,2,0)),
                                        ["original_image", "heat_map"],
                                        ["all", "positive"],
                                        cmap=default_cmap,
                                        show_colorbar=True)

    plt.savefig(save_path)


    torch.manual_seed(0)
    np.random.seed(0)

    gradient_shap = GradientShap(model)

    # Defining baseline distribution of images
    rand_img_dist = torch.cat([input * 0, input * 1])

    attributions_gs = gradient_shap.attribute(input,
                                            n_samples=50,
                                            stdevs=0.0001,
                                            baselines=rand_img_dist,
                                            target=pred_label_idx)

_ = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze(0).cpu().detach().numpy(), (1,2,0)),
                                      np.transpose(transformed_img.squeeze(0).cpu().detach().numpy(), (1,2,0)),
                                      ["original_image", "heat_map"],
                                      ["all", "absolute_value"],
                                      cmap=default_cmap,
                                      show_colorbar=True)

if __name__ == '__main__':
    

    print(f'torch version: {torch.__version__}')
    print(f'cuda version: {torch.version.cuda}')

    args = get_args()
    main(args)
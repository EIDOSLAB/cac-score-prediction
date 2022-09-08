import torch
import random 
import os
import numpy as np
import torchvision.transforms as transforms


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)


def get_transforms(img_size, crop, mean, std):
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    return train_transforms, test_transform


def local_copy(dataset, require_cac_score):
    if require_cac_score:
        return  [(dataset[j][0],dataset[j][1],dataset[j][2]) for j in range(len(dataset))]
    else:
        return [(dataset[j][0],dataset[j][1]) for j in range(len(dataset))]

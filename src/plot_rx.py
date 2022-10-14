import torch
import torch.nn.functional as F

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import torchvision
from torchvision import models
from torchvision import transforms
from PIL import Image
from dataset import CalciumDetection
from utility.utils import get_transforms
from models import  cac_detector
#from models import load_densenet_mlp, HierarchicalResidual
#from utility import dicom_img





def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()
    
    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)

    return new_img

"""
        path = self.elem[idx] + os.listdir(self.elem[idx])[0]
        dimg = pydicom.dcmread(path, force=True)

        img16 = apply_windowing(dimg.pixel_array, dimg)   
        img_eq = exposure.equalize_hist(img16)
        img8 = convert(img_eq, 0, 255, np.uint8)
        img_array = ~img8 if dimg.PhotometricInterpretation == 'MONOCHROME1' else img8
"""
from pydicom.pixel_data_handlers.util import  apply_windowing
from skimage import exposure
from PIL import Image
import pydicom

def main():
    path_data = "/src/dataset/"
    save_data = "/src/rx/"
    mean, std = [0.5024], [0.2898]
    transform, _ = get_transforms(img_size=1248, crop=1024, mean = mean, std = std)




    whole_dataset = CalciumDetection(path_data, transform, mode='classification', require_cac_score=True)


    #dataloader = torch.utils.data.DataLoader(whole_dataset, batch_size=1, shuffle=False)
    
    import os
    lista =whole_dataset.elem
    for i,c in enumerate(lista):
        print(c, " ",os.listdir(c))
        path = c + os.listdir(c)[0]
        
        dimg = pydicom.dcmread(path, force=True)
        img16 = apply_windowing(dimg.pixel_array, dimg)
        img_eq = exposure.equalize_hist(img16)
        img8 = convert(img_eq, 0, 255, np.uint8)
        img_array = ~img8 if dimg.PhotometricInterpretation == 'MONOCHROME1' else img8
        img = Image.fromarray(img_array)
        cc = str(whole_dataset.elem[i]).split("/")[-3]
if __name__ == "__main__":
    main()

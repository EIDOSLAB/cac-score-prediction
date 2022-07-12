import torch
from PIL import Image
import glob
import os
import torchvision
import pydicom
import numpy as np
import sqlite3

from pydicom.pixel_data_handlers.util import  apply_windowing, apply_modality_lut
from skimage import exposure

PATH_PLOT = '/home/fiodice/project/plot_training/'

## For wrong entry in site.db
def get_patient_id(dimg):
    if dimg.PatientID == 'CAC_1877':
        return dimg.PatientName
    else:
        return dimg.PatientID


def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()
    
    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)

    return new_img


class CalciumDetection(torch.utils.data.Dataset):
    def __init__(self, data_dir, labels_path, transform=None, require_path_file=False):
        self.root = data_dir
        self.elem = glob.glob(self.root + '*' + '/rx/')

        conn = sqlite3.connect(labels_path)
        conn.row_factory = sqlite3.Row  
        cursor = conn.cursor()

        self.labels = [dict(row) for row in cursor.execute('SELECT * FROM patient').fetchall()]
        self.transform = transform
        self.require_path_file = require_path_file


    def __len__(self):
        return len(self.elem)

    def __getitem__(self, idx):
        path = self.elem[idx] + os.listdir(self.elem[idx])[0]
        dimg = pydicom.dcmread(path, force=True)
        img16 = apply_windowing(dimg.pixel_array, dimg)   
        img_eq = exposure.equalize_hist(img16)
        img8 = convert(img_eq, 0, 255, np.uint8)
        img_array = ~img8 if dimg.PhotometricInterpretation == 'MONOCHROME1' else img8
        img = Image.fromarray(img_array)
        
        cac_score = [label for label in self.labels if label['id'] == get_patient_id(dimg)][0]['cac_score']
        label = 0 if int(cac_score) in range(0, 11) else 1

        if self.transform is not None:
            img = self.transform(img=img)
        else:
            img = torchvision.transforms.ToTensor()(img)

        if self.require_path_file:
            return img, path, label
        else:
            return img, label



class CalciumDetectionRegression(torch.utils.data.Dataset):
    def __init__(self, data_dir, labels_path, transform=None):
        self.root = data_dir
        self.elem = glob.glob(self.root + '*' + '/rx/')

        conn = sqlite3.connect(labels_path)
        conn.row_factory = sqlite3.Row  
        cursor = conn.cursor()

        self.labels = [dict(row) for row in cursor.execute('SELECT * FROM patient').fetchall()]
        self.cac_scores = np.array([patient['cac_score'] for patient in self.labels])
        self.transform = transform


    def __len__(self):
        return len(self.elem)

    def __getitem__(self, idx):
        # Process img                
        path = self.elem[idx] + os.listdir(self.elem[idx])[0]
        dimg = pydicom.dcmread(path, force=True)
        img16 = apply_windowing(dimg.pixel_array, dimg)
        img_eq = exposure.equalize_hist(img16)
        img8 = convert(img_eq, 0, 255, np.uint8)
        img_array = ~img8 if dimg.PhotometricInterpretation == 'MONOCHROME1' else img8
        img = Image.fromarray(img_array)

        # Process label                
        cac_score = [label for label in self.labels if label['id'] == get_patient_id(dimg)][0]['cac_score']
        #cac_clip = np.clip([cac_score],a_min=0, a_max=2000)
        #log_cac_score = np.log(cac_clip + 1)[0] 
        #cac_log = np.log((np.clip([cac_score],a_min=0, a_max=2000) + 0.001))
        #cac_norm = norm_log(cac_log)[0]

        if self.transform is not None:
            img = self.transform(img=img)
        else:
            img = torchvision.transforms.ToTensor()(img)

        return img.float(), cac_score 


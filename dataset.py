import torch
import glob
import os
import pydicom
import numpy as np
import sqlite3

from pydicom.pixel_data_handlers.util import  apply_windowing
from skimage import exposure
from PIL import Image


## For get the corrent entry in labels.db
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
    def __init__(self, data_dir, transform, mode, require_cac_score=False):
        self.root = data_dir
        self.elem = glob.glob(self.root + '*' + '/rx/')

        path_labels = data_dir + 'labels.db'

        conn = sqlite3.connect(path_labels)
        conn.row_factory = sqlite3.Row  
        cursor = conn.cursor()

        self.labels = [dict(row) for row in cursor.execute('SELECT * FROM patient').fetchall()]
        self.transform = transform
        self.mode = mode
        self.require_cac_score = require_cac_score


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
        
        if self.mode == 'regression':
            return img.float(), cac_score
        else:
            if self.require_cac_score:
                return img, label, cac_score
            else:
                return img, label
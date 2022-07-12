import os
from os import listdir
import sys
from pydicom import dcmread
import matplotlib.pyplot as plt
#from preprocessing import windowing, windowing_param
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_windowing, convert_color_space
from PIL.Image import fromarray

if __name__ == '__main__':
    # CAC = 0
    path_img1 = "/home/fiodice/project/dataset/CAC_400/rx/IM-0001-0001.dcm"
    path_img2 = "/home/fiodice/project/dataset/CAC_404/rx/IM-0008-0001-0002.dcm"
    plot_dir = '/home/fiodice/project/plot_training/'

    DCM_files = []

    img1 = dcmread(path_img1)
    img2 = dcmread(path_img2)

    f = plt.figure()
    ax1 = f.add_subplot(1, 2, 1)
    ax1.title.set_text('CAC score = 0')
    ax1.grid(False)
    plt.imshow(img1.pixel_array, cmap=plt.cm.gray)       
    plt.axis('off')

    ax1 = f.add_subplot(1, 2, 2)
    ax1.title.set_text('CAC score = 5824')
    ax1.grid(False)
    plt.imshow(img2.pixel_array, cmap=plt.cm.gray)
    plt.tight_layout()
    plt.axis('off')

    plt.savefig(plot_dir + 'sample_xray.png')
    
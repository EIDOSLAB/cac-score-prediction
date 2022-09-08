from unicodedata import name
from pip import main
import os
from pydicom import dcmread

import sqlite3

tag_view_position = ('0018','5101')

##### script bash for deleting all ct
# rm -rf `find . -type d -name ct`


def fix_labels(labels_path):
    conn = sqlite3.connect(labels_path)
    conn.row_factory = sqlite3.Row  
    cursor = conn.cursor()
    # Nelle immagini (29,30) l'id è CAC_030 mentre nel site.db è CAC_30 quindi aggiorno
    # nel site per poter matchare
    cursor.execute('''UPDATE patient SET id = 'CAC_030' WHERE id = 'CAC_30' ''')
    conn.commit()
    cursor.execute('''UPDATE patient SET id = 'CAC_029' WHERE id = 'CAC_29' ''')
    conn.commit()
    # Nelle immagini (83,96) il valore dell'id è CAC_0XX anche se l'immagine è in un cartella col
    # suffisso corr, ma nel site.db l'entry dell'id corretta è CAC_0XX_corr
    # quindi rimuovo quelle sbagliate CAC_0XX e aggiorno quelle corrette CAC_0XX_corr
    # con il valore dell'id corretto CAC_0XX 
    
    cursor.execute(''' DELETE FROM patient WHERE id = 'CAC_083' ''')
    cursor.execute(''' DELETE FROM patient WHERE id = 'CAC_096' ''')

    cursor.execute('''UPDATE patient SET id = 'CAC_083' WHERE id = 'CAC_083_corr' ''')
    conn.commit()
    cursor.execute('''UPDATE patient SET id = 'CAC_096' WHERE id = 'CAC_096_corr' ''')
    conn.commit()

    # Problema con 097: l'immagine ha id 098 e tale id non è presente nel site
    cursor.execute('''UPDATE patient SET id = 'CAC_098' WHERE id = 'CAC_097' ''')
    conn.commit()


def clean_data(path_data, remove_file):
    
    # Doppia radiografia frontale: 326 -> IM-0003-1003.dcm, IM-0002-1002.dcm 
    #                               439 -> IM-0104-1001.dcm, IM-0105-1002.dcm

    DCM_files = []
    for dir_name, _, file_list in os.walk(path_data):
        for filename in file_list:
            if ".dcm" in filename.lower():
                DCM_files.append(os.path.join(dir_name, filename))

    print("Number of (.dcm) files =", len(DCM_files))

    count_front = 0
    count_metadata = 0
    count_lat = 0
    count_empty = 0
    count_img_with_no_position = 0

    for path_file in DCM_files:
        ds = dcmread(path_file, force=True)

        # If there isn't PixelData in DICOM file remove it
        if 'PixelData' not in ds:
            if remove_file:
                os.remove(path_file) 
        else:
            size = os.path.getsize(path_file)

            if tag_view_position in ds:
                view_position = ds[('0018','5101')].value

                if view_position == '':
                    count_empty += 1
                elif view_position == 'PA' or view_position == 'AP' or view_position == 'NN':
                    count_front += 1
                # remove lateral rx
                elif view_position == 'LL' or view_position == 'LAT' or view_position == 'LATERALE':
                    count_lat += 1
                    if remove_file:
                        os.remove(path_file) 
                else:
                # if view_position not in DICOM file is not an image, so remove
                    count_img_with_no_position += 1
            else:
                # if the file is less the 1MB is a metedata file
                if size < 1000000:
                    count_metadata += 1
                
                if remove_file:
                        os.remove(path_file) 

                    

    tot = count_img_with_no_position + count_empty + count_front
    print(f'Metadata {count_metadata}\n RX PA {count_front}\n RX LAT {count_lat}\n RX Empty position {count_empty}')
    print(f'File to check by hand {tot}')


if __name__ == '__main__':
    path_data = '/home/fiodice/project/original_data'

    clean_data(path_data, False)
    
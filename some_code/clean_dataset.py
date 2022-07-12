from unicodedata import name
from pip import main
import os
from pydicom import dcmread

import sqlite3

slope = ('0028','1053')     
intercept = ('0028','1052') 

def fixLabels():
    #labels_path = '/home/fiodice/project/data_custom/site.db'
    labels_path = ''
    conn = sqlite3.connect(labels_path)
    conn.row_factory = sqlite3.Row  
    cursor = conn.cursor()
    # Updating correct id 
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

    # Check
    labels = [dict(row) for row in cursor.execute('SELECT * FROM patient').fetchall()]



def removeFiles():
        #path_dicom = "/home/fiodice/project/data_custom/"
    path_dicom = "/home/fiodice/project/data_only_new/"

    
    # Doppia radiografia frontale: 326 -> IM-0003-1003.dcm, IM-0002-1002.dcm 
    #                               439 -> IM-0104-1001.dcm, IM-0105-1002.dcm

    # File of dicom image
    DCM_files = []
    for dir_name, sub_dir_list, file_list in os.walk(path_dicom):
        for filename in file_list:
            if ".dcm" in filename.lower():
                DCM_files.append(os.path.join(dir_name, filename))

    print("Number of (.dcm) files =", len(DCM_files))

    count_front = 0
    count_meta = 0
    count_lat = 0
    count_empty = 0
    count_img_with_no_position = 0

    for path_file in DCM_files:
        ds = dcmread(path_file, force=True)
        if 'PixelData' not in ds:
                print('removed')
                os.remove(path_file) 
        else:

            #if(slope in ds and intercept in ds):
            #    print(ds[slope].value, ds[intercept].value)
        
            size = os.path.getsize(path_file)
            print(f'Path file {path_file} Size {size}')
            # ('0018','5101') -> View Position 

            if ('0018','5101') in ds:
                view_position = ds[('0018','5101')].value

                if view_position == '':
                    count_empty += 1
                elif view_position == 'PA' or view_position == 'AP' or view_position == 'NN':
                    count_front += 1
                elif view_position == 'LL' or view_position == 'LAT' or view_position == 'LATERALE':
                    count_lat += 1
                    os.remove(path_file) 
            else:

                if 'PixelData' in ds:
                    count_img_with_no_position += 1
                else:
                    count_meta += 1
                    os.remove(path_file) 
                

        tot = count_img_with_no_position + count_empty + count_front
        print(f'Meta {count_meta} PA {count_front} LAT {count_lat} Empty {count_empty} Fault {count_img_with_no_position}')
        print(f'File to check {tot}')


if __name__ == '__main__':
    path_labels =  '/home/fiodice/project/labels/site.db'
    conn = sqlite3.connect(path_labels)
    conn.row_factory = sqlite3.Row  
    cursor = conn.cursor()

    cursor.execute(''' DELETE FROM patient WHERE id = 'CAC_083' ''')
    cursor.execute(''' DELETE FROM patient WHERE id = 'CAC_096' ''')
    #cursor.execute('''UPDATE patient SET id = 'CAC_187' WHERE id = 'CAC_083_corr' ''')


    cursor.execute('''UPDATE patient SET id = 'CAC_083' WHERE id = 'CAC_083_corr' ''')
    conn.commit()
    cursor.execute('''UPDATE patient SET id = 'CAC_096' WHERE id = 'CAC_096_corr' ''')
    conn.commit()

    # Problema con 097: l'immagine ha id 098 e tale id non è presente nel site
    cursor.execute('''UPDATE patient SET id = 'CAC_098' WHERE id = 'CAC_097' ''')
    conn.commit()
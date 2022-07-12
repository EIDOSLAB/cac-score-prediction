# DeepLearning CalciumDetection


Coronary artery calcifications are often overlooked in routine clinical practice, and even for an experienced radiologist it is difficult to detect calcium from an X-ray image by eye compared to a CT scan.
Chest x-rays are obtained much more commonly than CT scans at a much lower cost; for this reason an automatic radiographic image analysis system could be of fundamental importance for the early detection of calcium.
Through this work we aim to build a process, based on convolutional neural networks, capable of predicting the presence or absence of coronary calcium from chest radiographs, evaluating the performance in terms of accuracy obtained.


### Data

For the correct functioning of the code shown in this repo it is necessary to work with the folder `/home/fiodice/project/dataset` in escher.
The code in `project/src/some_code/clean_dataset.py` is a script for cleaning the dataset, but many labels of the dicom files are not populated. The script removes most of the useless files, but a manual cleanup was performed for the rest.

### Training

Two approaches for solving the task were performed, for both approaches a cross validation with 5 fold was performed. For the files that I am going to list you can specify: 

- `path_data` : path to folder with the data
- `path_labels` : path to labels
- `path_model` : path to pretrained model

##### Classification
- In `cross_clf` the classification approach is implemented

##### Regression

-  In `cross_regr.py` the regression approach is implemented 
-  In `cross_regr_str_kfold.py` is implemented a cross-validation stratified

##### Result

The models resulting from the training could be found on `project/src/models_pt/final`. More details on the methodology and quality of the results can be found in the thesis of Francesco Iodice followed by Professor Marco Grangetto and the co-supervisors Alberto Presta and Carlo Alberto Barbano.
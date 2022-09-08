# DeepLearning CalciumDetection


Coronary artery calcifications are often overlooked in routine clinical practice, and even for an experienced radiologist it is difficult to detect calcium from an X-ray image by eye compared to a CT scan.
Chest x-rays are obtained much more commonly than CT scans at a much lower cost; for this reason an automatic radiographic image analysis system could be of fundamental importance for the early detection of calcium.
Through this work we aim to build a process, based on convolutional neural networks, capable of predicting the presence or absence of coronary calcium from chest radiographs, evaluating the performance in terms of accuracy obtained.


### Data

For the correct functioning of the code shown in this repo it is necessary to work with the dataset content in the folder `/home/fiodice/project/dataset`.
The code in `script_clean_data/clean_dataset.py` is a script for cleaning the dataset, but many labels of the dicom files are not populated. The script removes most of the useless files, but a manual cleanup was performed for the rest.

### Training

Two approaches were performed for solving the task, for both approaches a cross validation with 5 fold was performed. 

-  In `train_cac_classifier.py` the classification approach is implemented
-  In `train_cac_regressor.py` the regression approach is implemented

For the files you can specify the following parameters: 

- `epochs` : num. of epochs (default 50)
- `lr` : learning rate default (3e-4) for classification and (1e-3) for regression
- `arch` : encoder architecture (densenet121 or resnet18 or efficientNet)
- `viz` : save images of metrics and losses
- `save` : save model
- `wd` : weight decay value (default 1e-4)
- `batchsize` : samples for batch 
- `momentum` : momentum value (default 0.9)
- `kfold` : folds for cross-validation (default 5)


##### Result

The best models resulting with the relevant training details, could be found in the result `models_cac_pt`. More details on the methodology and quality of the results can be found in the thesis of Francesco Iodice followed by Professor Marco Grangetto and the co-supervisors Alberto Presta and Carlo Alberto Barbano.
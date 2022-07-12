
import torch
import torch.nn as nn
import sys

from models import model_HR

sys.path.insert(0, '/home/fiodice/project/src')


def load_densenet(path_model):
    model = model_HR.HierarchicalResidual(encoder='densenet121')
    dict_model = torch.load(path_model)["model"]
    model.load_state_dict(dict_model)

    del model.fc1
    del model.fc2

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(1024, 2)

    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model



def test_calcium_det(path_model):
    model = model_HR.HierarchicalResidual(encoder='densenet121')
    dict_model = torch.load(path_model)["model"]

    del model.fc1
    del model.fc2

    model.fc =  torch.nn.Sequential(
            torch.nn.Linear(1024, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2))
        
    model.load_state_dict(dict_model)

    return model


def densenet_classifier(path_model):
    model = model_HR.HierarchicalResidual(encoder='densenet121')
    dict_model = torch.load(path_model)["model"]
    model.load_state_dict(dict_model)

    del model.fc1
    del model.fc2

    for param in model.parameters():
        param.requires_grad = False

    model.fc =  torch.nn.Sequential(
            torch.nn.Linear(1024, 64),
            #torch.nn.Dropout(p=0.4),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2))

    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model


def load_resnet_mlp(path_model):
    model = model_HR.HierarchicalResidual(encoder='resnet18')
    dict_model = torch.load(path_model)["model"]
    model.load_state_dict(dict_model)

    del model.fc1
    del model.fc2

    for param in model.parameters():
        param.requires_grad = False

    model.fc =  torch.nn.Sequential(
            torch.nn.Linear(512, 64),
            #torch.nn.Dropout(p=0.4),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2))

    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model


def load_effcientNet(path_model):
    model = model_HR.HierarchicalResidual(encoder='efficientnet-b0')
    dict_model = torch.load(path_model)["model"]
    model.load_state_dict(dict_model)

    for param in model.parameters():
        param.requires_grad = False

    del model.fc1
    del model.fc2

    model.fc =  torch.nn.Sequential(
            torch.nn.Linear(1280, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1))

    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model


def densenet_regressor(path_model):
    model = model_HR.HierarchicalResidual(encoder='densenet121')
    dict_model = torch.load(path_model)["model"]
    model.load_state_dict(dict_model)

    del model.fc1
    del model.fc2

    for param in model.parameters():
        param.requires_grad = False

    model.fc =  torch.nn.Sequential(
            torch.nn.Linear(1024, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1))
    

    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model


def test_densenet_regressor(path_model):
    model = model_HR.HierarchicalResidual(encoder='densenet121')
    dict_model = torch.load(path_model)["model"]

    del model.fc1
    del model.fc2

    model.fc =  torch.nn.Sequential(
            torch.nn.Linear(1024, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1))
    
    model.load_state_dict(dict_model)
    return model


## unfreeze last layer backbone

def unfreeze_param_lastlayer_dense_regr(model):
    #model_last_layer = model.encoder[-3][-2].denselayer16
    # adding dropout
    model_last_layer = model.encoder[-4][-2].denselayer16

    for param in model_last_layer.parameters():
        param.requires_grad = True

    return model_last_layer

def unfreeze_param_lastlayer_dense_clf(model):
    #model_last_layer = model.encoder[-3][-2].denselayer16
    # adding dropout
    model_last_layer = model.encoder[-4][-2].denselayer16

    for param in model_last_layer.parameters():
        param.requires_grad = True

    return model_last_layer


def unfreeze_param_last2layer_dense(model):
    model_lastblock_layer16 = model.encoder[-3][-2].denselayer16
    model_lastblock_layer15 = model.encoder[-3][-2].denselayer15

    for param in model_lastblock_layer16.parameters():
        param.requires_grad = True

    for param in model_lastblock_layer15.parameters():
        param.requires_grad = True

    return model_lastblock_layer16, model_lastblock_layer15


def unfreeze_param_lastlayer_eff(model):
    model_last_layer = list(model.encoder.children())[-5]

    for i, layer in enumerate((list(model.encoder.children()))):
        print(f'For {i} layer {layer}')

    for param in model_last_layer.parameters():
        param.requires_grad = True

    return model_last_layer


def unfreeze_param_lastlayer_res(model):
    model_last_layer = model.encoder[-2][-1]

    for param in model_last_layer.parameters():
        param.requires_grad = True

    return model_last_layer
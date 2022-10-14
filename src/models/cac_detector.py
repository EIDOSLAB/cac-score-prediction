import torch
import sys

from models.model_chexpert import HierarchicalResidual

sys.path.insert(0, '/home/fiodice/project/src')


def unfreeze_lastlayer_encoder(model, encoder_name):
    encoder_last_layer = None
    
    if encoder_name == 'densenet121':
        encoder_last_layer = model.encoder[-3][-2].denselayer16
    elif encoder_name == 'resnet18':
        encoder_last_layer = model.encoder[-2][-1]
    elif encoder_name == 'efficientNet':
        encoder_last_layer = list(model.encoder.children())[-5]

    for param in encoder_last_layer.parameters():
        param.requires_grad = True

    return encoder_last_layer


def load_model(path_model, path_encoder, mode):
    model = CalciumDetector(encoder = 'densenet121', path_encoder = path_encoder, mode=mode)
    dict_model = torch.load(path_model)["model"]
    model.load_state_dict(dict_model)
    return model


class CalciumDetector(torch.nn.Module):
    def __init__(self, path_encoder, encoder, mode = 'classifier'):
        super().__init__()

        if encoder == 'densenet121':
            size_fc = 1024
        elif encoder == 'resnet18':
            size_fc = 512 
        elif encoder == 'efficientnet-b0':
            size_fc = 1280
        else:
            print(f'Unkown encoder value: {encoder}')
            exit(1)
            
        model_chexpert = HierarchicalResidual(encoder = encoder)
        dict_model = torch.load(path_encoder)["model"]
        model_chexpert.load_state_dict(dict_model)

        del model_chexpert.fc1
        del model_chexpert.fc2

        for param in model_chexpert.parameters():
            param.requires_grad = False

        self.encoder = model_chexpert.encoder
        self.fc = None

        if mode == 'classifier':
            self.fc =  torch.nn.Sequential(
                    torch.nn.Linear(size_fc, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, 2))
        elif mode == 'regressor':
            self.fc =  torch.nn.Sequential(
                    torch.nn.Linear(size_fc, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, 1))

        for param in self.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
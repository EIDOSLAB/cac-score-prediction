import torch
import torchvision
from efficientnet_pytorch import EfficientNet


class HierarchicalResidual(torch.nn.Module):
    def __init__(self, encoder='resnet18', pretrained=True):
        super().__init__()

        self.encoder_name = encoder
        self.encoder = None
        self.num_ft = 0

        if 'resnet' in encoder:
            resnets = {
                'resnet18': torchvision.models.resnet18,
                'resnet34': torchvision.models.resnet34,
                'resnet50': torchvision.models.resnet50,
                'resnet101': torchvision.models.resnet101,
            }

            self.encoder = resnets[encoder](pretrained=pretrained)
            self.encoder.conv1.weight.data = self.encoder.conv1.weight.data[:, :1]
            self.encoder.conv1.in_channels = 1
            self.num_ft = self.encoder.fc.in_features
            self.encoder = torch.nn.Sequential(*list(self.encoder.children())[:-1])
        
        elif 'densenet' in encoder:
            self.encoder = torch.hub.load('pytorch/vision:v0.6.0', encoder, pretrained=pretrained)
            self.encoder.features.conv0.weight.data = self.encoder.features.conv0.weight.data[:, :1]
            self.encoder.features.conv0.in_channels = 1
            self.num_ft = self.encoder.classifier.in_features
            self.encoder = torch.nn.Sequential(
                self.encoder.features,
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout2d(p=0.5),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
            )

        elif 'efficientnet' in encoder:
            self.encoder = EfficientNet.from_pretrained(encoder)
            self.encoder._conv_stem.weight.data = self.encoder._conv_stem.weight.data[:, :1]
            self.encoder._conv_stem.in_channels = 1
            #self.encoder = EfficientNet.from_pretrained(encoder)
            self.num_ft = self.encoder._fc.in_features

        elif 'resnext' in encoder: 
            self.encoder = torch.hub.load('pytorch/vision:v0.8.0', encoder, pretrained=True)
            self.encoder.conv1.weight.data = self.encoder.conv1.weight.data[:, :1]
            self.encoder.conv1.in_channels = 1
            self.num_ft = self.encoder.fc.in_features
            self.encoder = torch.nn.Sequential(*list(self.encoder.children())[:-1])
        else:
            print(f'Unkown encoder {encoder}')
            exit(1)

        """parent classes [
            No Finding, Enlarged Cardiomediastinum, Lung Opacity, 
            Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support devices
        ]
        """
        self.fc1 = torch.nn.Linear(in_features=self.num_ft, out_features=8, bias=True)

        """child classes [
            Cardiomegaly, Lung Lesion, Edema, Consolidation, Pneumonia, Atelactasis
        ]
        """
        self.fc2 = torch.nn.Linear(in_features=self.num_ft+8, out_features=6, bias=True)
        
        # Sort output with correct label order
        output_order = torch.tensor([2, 4, 5, 6, 7, 8, 0, 1, 3, 9, 10, 11, 12, 13])
        self.out_idx = torch.argsort(output_order)
    
    def forward(self, x):
        if 'efficientnet' in self.encoder_name:
            x = self.encoder.extract_features(x)
            x = self.encoder._avg_pooling(x)
        else:
            x = self.encoder(x)

        # correction from original code here
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


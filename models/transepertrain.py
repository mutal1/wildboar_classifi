import torch
import torch.nn as nn
import timm

class TransferTrainModel(nn.Module):
    def __init__(self):
        super(TransferTrainModel, self).__init__()
        # Load the pretrained Xception model from timm
        self.xception = timm.create_model('xception', pretrained=True)
        
        # Replace the last fully connected layer with a new one for the binary classification task
        num_features = self.xception.fc.in_features
        self.xception.fc = nn.Linear(num_features, 2)
        self.sigmoid = nn.Sigmoid()
        # Freeze the pretrained layers #fine turning
        # for param in self.xception.parameters():
            # param.requires_grad = True
        
        # for name, param in self.xception.named_parameters():
        #    if name != 'fc.weight' and name != 'fc.bias':
        #        param.requires_grad = False

    def forward(self, x):
        x = torch.cat([x, x, x], dim=1)
        x = self.xception(x)
        x = self.sigmoid(x)
        return x
    
# import torch
# import torch.nn as nn
# import torchvision.models as models

# class TransferTrainModel(nn.Module):
#     def __init__(self):
#         super(TransferTrainModel, self).__init__()
#         # Load the pretrained ResNet-50 model
#         self.resnet = models.resnet50(pretrained=True)
        
#         # Replace the last fully connected layer
#         num_features = self.resnet.fc.in_features
#         self.resnet.fc = nn.Linear(num_features, 2)
        
#         # Freeze the pretrained layers
#         for name, param in self.resnet.named_parameters():
#            if name != 'fc.weight' and name != 'fc.bias':
#                param.requires_grad = False


#     def forward(self, x):
#         x = torch.cat([x, x, x], dim=1)
#         x = self.resnet(x)
#         return x

# import torch
# import torch.nn as nn
# import torchvision.models as models

# class TransferTrainModel(nn.Module):
#     def __init__(self):
#         super(TransferTrainModel, self).__init__()

#         # Load a smaller variant of the pre-trained model
#         self.resnet = models.resnet18(pretrained=True)

#         # Replace the last fully connected layer with a new one for your classification task
#         num_features = self.resnet.fc.in_features
#         self.resnet.fc = nn.Linear(num_features, 2)

#         # Freeze the pre-trained layers
#         for name, param in self.resnet.named_parameters():
#             if name != 'fc.weight' and name != 'fc.bias':
#                 param.requires_grad = False
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = torch.cat([x, x, x], dim=1)
#         x = self.resnet(x)
#         x = self.sigmoid(x)
#         return x

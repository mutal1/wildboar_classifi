import torch
import torch.nn as nn
import torch.nn.functional as F

class GLU(nn.Module):
    def __init__(self, input_num):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(x)
        res = lin * sig
        return res

VGG16_layers = [16, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']


class VGG_16(nn.Module):
    def __init__(self, in_channels = 1):
        super(VGG_16, self).__init__()
        self.in_channels = 1
        self.conv_layers = self.generate_conv_layers(VGG16_layers)
        self.fc_layers = nn.Sequential(
                    nn.Linear(512 , 4096),
                    nn.ReLU(),
                    nn.Linear(4096, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, 2),
                    nn.Sigmoid()
                )

    def generate_conv_layers(self, layer_architecture):
        layers = []
        in_channels = self.in_channels


        for x in layer_architecture:
            if type(x) == int:
                out_channels = x
                layers += [#dw
                           nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = (3, 3), stride = (1, 1), padding='same',groups=in_channels),
                           nn.BatchNorm2d(x),
                           GLU(out_channels),
                           #pw
                        #    nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size=1,),
                        #    nn.BatchNorm2d(x, eps=0.001, momentum=0.99),
                        #    GLU(out_channels),
                        #    nn.Dropout(0.5)
                        ]
                in_channels = x
            else:
                layers += [nn.AvgPool2d(kernel_size = (2, 2), stride = (2, 2))]

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv_layers(x)
        _,_,freq,frame = x.shape
        pool_layer = nn.AvgPool2d(kernel_size=(freq,frame) , stride=(1,1))
        x = pool_layer(x)
        x = x.squeeze(-1,-2)
        x = self.fc_layers(x)

        return x

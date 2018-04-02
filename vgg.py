import torch
import torch.nn as nn
import math
import sys

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg(vgg_type):

    # 11
    if vgg_type == 'vgg11':
        model = VGG(make_layers(cfg['A']))
    elif vgg_type == 'vgg11_bn':
        model = VGG(make_layers(cfg['A'], batch_norm=True))

    # 13
    elif vgg_type == 'vgg13':
        model = VGG(make_layers(cfg['B']))
    elif vgg_type == 'vgg13_bn':
        model = VGG(make_layers(cfg['B'], batch_norm=True))

    # 16
    elif vgg_type == 'vgg16':
        model = VGG(make_layers(cfg['D']))
    elif vgg_type == 'vgg16_bn':
        model = VGG(make_layers(cfg['D'], batch_norm=True))

    # 19
    elif vgg_type == 'vgg19':
        model = VGG(make_layers(cfg['E']))
    elif vgg_type == 'vgg19_bn':
        model = VGG(make_layers(cfg['E'], batch_norm=True))

    else:
        print('wrong model name')

    return model

def main():
    batch = 1
    input_var = torch.FloatTensor(batch, 3, 224, 224)
    input_var = torch.autograd.Variable(input_var.cuda(), volatile=True)

    model = vgg(sys.argv[1]).cuda()

    #run inference
    model.eval()
    output = model(input_var)

    print('finish inference ' + sys.argv[1])

if __name__ == '__main__':
    main()
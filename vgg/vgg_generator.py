import torch
import torch.nn as nn
import math
import sys

dict_index = int(sys.argv[1])

class VGG(nn.Module):

    #define the network
    def __init__(self, features, classifier):
        super(VGG, self).__init__()
        self.tag = nn.Softplus()

        #conv layer
        self.features = features

        #fc layer
        self.classifier = classifier

        self._initialize_weights()

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
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    #define forward
    def forward(self, x):
        x = self.tag(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.tag(x)
        return x


def make_conv(cfg):
    layers = []
    layer_c= cfg['# of layers'][0]
    if_bn  = cfg['bn']
    if_bias= cfg['bias']

    c = 1
    while c < layer_c:
        #conv
        pc = 'pc_'+str(c)
        if pc in cfg['net']:
            kw = cfg['net'][pc]['kw']
            sw = cfg['net'][pc]['sw']
            pw = cfg['net'][pc]['pw']
            kh = cfg['net'][pc]['kh']
            sh = cfg['net'][pc]['sh']
            ph = cfg['net'][pc]['ph']

            cin  = cfg['net'][pc]['cin']
            cout = cfg['net'][pc]['cout']

            conv2d = nn.Conv2d(cin, cout, kernel_size=(kh, kw), stride=(sh, sw), padding=(ph, pw), bias=if_bias)

            if if_bn:
                layers += [conv2d, nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]

        #pooling
        pp = 'pp_'+str(c)
        if pp in cfg['net']:
            if cfg['net'][pp]['type'] == 'max':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif cfg['net'][pp]['type'] == 'avg':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]

        c += 1

    return nn.Sequential(*layers)

def make_fc(cfg):
    layers = []
    layer_f= cfg['# of layers'][1]

    f = 1
    while f < layer_f:
        pf = 'pf_'+str(f)
        if f == layer_f-1:
            layers += [nn.Linear(cfg['net'][pf]['cin'], cfg['net'][pf]['cout'])]
        else:
            layers += [nn.Linear(cfg['net'][pf]['cin'], cfg['net'][pf]['cout']), nn.ReLU(True)]

        f += 1

    return nn.Sequential(*layers)

def vgg(model_graph):
    return VGG(make_conv(model_graph), make_fc(model_graph))


def main():
    #initial input
    batch = 1
    input_var = torch.FloatTensor(batch, 3, 224, 224)
    input_var = torch.autograd.Variable(input_var.cuda(), volatile=True)

    '''
    for i in range(10):
        f = open('./graph/VGG' + str(dict_index) + '.txt', 'r')
        a = f.read()
        cfg = eval(a)
        f.close()

        model = vgg(cfg).cuda()

        #run inference
        model.eval()
        output = model(input_var)

        print('finish inference: ' + str(i))
    '''

    # load dictionary
    f = open('./graph/VGG' + str(dict_index) + '.txt', 'r')
    a = f.read()
    cfg = eval(a)
    f.close()

    model = vgg(cfg).cuda()

    # run inference
    model.eval()
    output = model(input_var)


    print('finish inference: ' + str(dict_index))


    return 0


if __name__ == '__main__':
    main()


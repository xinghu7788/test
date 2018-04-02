import sys
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg(model_name):
    VGG_model = {'label':[], 'para':[]}
    vgg_type = model_name

    if_bn = False
    if len(vgg_type) > 5:
        if_bn = True

    if vgg_type == 'vgg11' or vgg_type == 'vgg11_bn':
        sequence = cfg['A']
    elif vgg_type == 'vgg13' or vgg_type == 'vgg13_bn':
        sequence = cfg['B']
    elif vgg_type == 'vgg16' or vgg_type == 'vgg16_bn':
        sequence = cfg['D']
    elif vgg_type == 'vgg19' or vgg_type == 'vgg19_bn':
        sequence = cfg['E']

    c_in = 3
    f    = 224
    #conv and pooling layers
    for i in range(len(sequence)):
        v = sequence[i]

        #pooling layer
        if v == 'M':
            VGG_model['label'].append('pooling')
            VGG_model['para'].append([2, 2, 0, 'max'])#[k, s, p, type]
            f /= 2

        #conv
        else:
            #conv
            VGG_model['label'].append('conv')
            VGG_model['para'].append([3, 3, 1, 1, 1, 1, c_in, v])#[kw, kh, sw, sh, pw, ph, cin, cout]
            #bn
            if if_bn:
                VGG_model['label'].append('bn')
                VGG_model['para'].append([v])
            #relu
            VGG_model['label'].append('relu')
            VGG_model['para'].append([])

            c_in = v


    # fc layer
    VGG_model['label'].append('fc')
    VGG_model['para'].append([512 * 7 * 7, 4096])
    VGG_model['label'].append('relu')
    VGG_model['para'].append([])

    VGG_model['label'].append('fc')
    VGG_model['para'].append([4096, 4096])
    VGG_model['label'].append('relu')
    VGG_model['para'].append([])

    VGG_model['label'].append('fc')
    VGG_model['para'].append([4096, 1000])

    return VGG_model

def main():
    VGG_model = vgg(sys.argv[1])
    print(VGG_model)

if __name__ == '__main__':
    main()



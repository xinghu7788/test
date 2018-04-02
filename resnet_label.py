'''
in resnet all parameters do not have bias
'''

import sys

cfg = {'resnet18':[2, 2, 2, 2, 'basicblock'], 'resnet34':[3, 4, 6, 3, 'basicblock'], 'resnet50':[3, 4, 6, 3, 'bottleneck'],
       'resnet101':[3, 4, 23, 3, 'bottleneck'], 'resnet152':[3, 8, 36, 3, 'bottleneck']}

expansion = {'basicblock':1, 'bottleneck':4}

def resnet_block(f_in, block_name, inplanes, planes, stride=1, downsample={}):
    block = {'label':[], 'para':[]}

    if block_name == 'basicblock':
        #conv1
        block['label'].append('conv')
        block['para'].append([3, 3, stride, stride, 1, 1, inplanes, planes])#[kw, kh, sw, sh, pw, ph, cin, cout]
        f_in /= stride
        block['label'].append('bn')
        block['para'].append([planes])
        block['label'].append('relu')
        block['para'].append([])

        #conv2
        block['label'].append('conv')
        block['para'].append([3, 3, 1, 1, 1, 1, planes, planes])
        block['label'].append('bn')
        block['para'].append([planes])


    elif block_name == 'bottleneck':
        #conv1
        block['label'].append('conv')
        block['para'].append([1, 1, 1, 1, 0, 0, inplanes, planes])
        block['label'].append('bn')
        block['para'].append([planes])
        block['label'].append('relu')
        block['para'].append([])

        #conv2
        block['label'].append('conv')
        block['para'].append([3, 3, stride, stride, 1, 1, planes, planes])
        f_in /= stride
        block['label'].append('bn')
        block['para'].append([planes])
        block['label'].append('relu')
        block['para'].append([])

        #conv3
        block['label'].append('conv')
        block['para'].append([1, 1, 1, 1, 0, 0, planes, planes * 4])
        block['label'].append('bn')
        block['para'].append([planes * 4])

    if downsample != {}:
        block['label'] += downsample['label']
        block['para'] += downsample['para']

    block['label'].append('add')
    block['para'].append([])
    block['label'].append('relu')
    block['para'].append([])

    return block

def make_layer(inplanes, f_in, block, planes, blocks, stride=1):
    layer = {'label':[], 'para':[]}
    downsample = {}
    if stride != 1 or planes * expansion[block] != inplanes:
        downsample = {'label':[], 'para':[]}
        downsample['label'].append('conv')
        downsample['para'].append([1, 1, stride, stride, 0, 0, inplanes, planes * expansion[block]])
        downsample['label'].append('bn')
        downsample['para'].append([planes * expansion[block]])


    #write first block
    current_block = resnet_block(f_in, block, inplanes, planes, stride, downsample)
    layer['label'] += current_block['label']
    layer['para']  += current_block['para']
    inplanes = planes * expansion[block]

    #write rest blocks
    for i in range(1, blocks):
        current_block = resnet_block(f_in, block, inplanes, planes)
        layer['label'] += current_block['label']
        layer['para']  += current_block['para']

    return layer



def resnet(model_name):
    resnet_model = {'label':[], 'para':[]}
    sequence = cfg[model_name]

    #write first conv and pool
    resnet_model['label'].append('conv')
    resnet_model['para'].append([7, 7, 2, 2, 3, 3, 3, 64])
    resnet_model['label'].append('bn')
    resnet_model['para'].append([64])
    resnet_model['label'].append('relu')
    resnet_model['para'].append([])

    resnet_model['label'].append('pooling')
    resnet_model['para'].append([3, 2, 1, 'max'])#[k, s, p, category]

    #resnet layers
    block = make_layer(64, 56, sequence[4], 64, sequence[0])
    resnet_model['label'] += block['label']
    resnet_model['para']  += block['para']

    block = make_layer(256, 56, sequence[4], 128, sequence[1], stride=2)
    resnet_model['label'] += block['label']
    resnet_model['para']  += block['para']

    block = make_layer(512, 28, sequence[4], 256, sequence[2], stride=2)
    resnet_model['label'] += block['label']
    resnet_model['para']  += block['para']

    block = make_layer(1024, 14, sequence[4], 512, sequence[3], stride=2)
    resnet_model['label'] += block['label']
    resnet_model['para']  += block['para']

    #avgpool and fc
    resnet_model['label'].append('pooling')
    resnet_model['para'].append([7, 1, 0, 'avg'])
    resnet_model['label'].append('fc')
    resnet_model['para'].append([2048, 1000])

    return resnet_model


def main():
    Resnet_model = resnet(sys.argv[1])
    print(Resnet_model)

if __name__ == '__main__':
    main()


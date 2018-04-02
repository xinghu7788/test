'''
VGG model generator
global: bn, bias

conv: P(asymmetric)=0.3 P(symmetric)=0.7
    1. kernel [1, 3, 5, 7] P(k) = 0.25
    2. padding[0, 1, 2, 3] (corresponding to kernel size) P(pi|ki)=1 P(pi|kj)=0
    3. stride [1, 2] P(s=1)=0.8 P(s=2)=0.2
    4. P(channel=U(64, 512))=0.3 P(channel=[64, 128, 256, 512])=0.7 P(i|channel=[64, 128, 256, 512])=0.25

pooling:
    1. kernel [2]
    2. padding[0]
    3. stride [2]
    4. P(type='avg')=P(type='max')=0.5

fc:
    1. channel [1024, 2048, 4096] P(c)=1/3
    P(end)=0.7

generation rule:
    if feature_size(width or height) < 7:
        P(fc)=0.7
        P(conv)=0.2
        P(pooling)=0.1
    else
        P(conv)=0.8
        P(pooling)=0.2

whole graph description: model{'class':'vgg', 'bn':True/False, 'bias':True/False, '# of layers':[layer_c, layer_f], 'net':{}}
conv feature map: model['net']['mc_'+str(layer_c)]{'mw': ,'mh': ,'c': }
conv parameter:   model['net']['pc_'+str(layer_c)]{'kw': ,'kh': ,'sw': ,'sh': ,'pw': ,'ph': ,'cin': ,'cout': }
pooling parameter:model['net']['pp_'+str(layer_c)]{'k': ,'s': ,'p': ,'type': }
fc feature map:   model['net']['mf_'+str(layer_f)]{'c': }
fc parameter:     model['net']['pf_'+str(layer_f)]{'cin': ,'cout': }
'''

import random
import math
import sys

klist = [1, 3, 5, 7]
plist = [0, 1, 2, 3]
slist = [1, 2]
clist = [64, 128, 256, 512]

def main():
    for i in range(int(sys.argv[1])):
        VGG_model = gen_VGG()

        #save vgg dictionary
        f = open('./graph/VGG'+str(i)+'.txt', 'w')
        f.write(str(VGG_model))
        f.close()

#get a random number from 0 to 1
def get_random():
    return random.uniform(0,1)

#conv layer (generate current pc and next mc)
def gen_conv(map, layer_c):
    gen_succ = False
    layer = {}

    # select the stride, kernel size, padding size, cout
    sw = int(get_random() // 0.8)
    kw = random.randint(0, 3)

    if get_random() >= 0.7:
        sh = int(get_random() // 0.8)
        kh = random.randint(0, 3)
    else:
        sh = sw
        kh = kw

    if get_random() >= 0.7:
        cout = random.randint(64, 512)
    else:
        cout = clist[random.randint(0, 3)]

    para = {'kw': klist[kw], 'kh': klist[kh], 'sw': slist[sw], 'sh': slist[sh], 'pw': plist[kw], 'ph': plist[kh],
                     'cin': map['c'], 'cout': cout}

    # write feature map
    mc = {}
    #int(math.floor((f_in + 2 * p - k) / s + 1))
    mc['mw'] = int(math.floor((map['mw'] + 2 * plist[kw] - klist[kw]) / slist[sw]+1))
    mc['mh'] = int(math.floor((map['mh'] + 2 * plist[kh] - klist[kh]) / slist[sh]+1))
    mc['c']  = para['cout']

    if mc['mw'] > 0 and mc['mh'] > 0:
        gen_succ = True
        layer['pc_'+str(layer_c)]   = para
        layer['mc_'+str(layer_c+1)] = mc

    return gen_succ, layer

#pooling layer (generate current pp and next mc)
def gen_pool(map, layer_c):
    gen_succ = False
    layer = {}

    mc = {}
    mc['mw'] = int(math.floor(map['mw']/2))
    mc['mh'] = int(math.floor(map['mh']/2))
    mc['c']  = map['c']


    if mc['mw'] > 0 and mc['mh'] > 0:
        gen_succ = True
        type = 'max'
        if get_random() >= 0.5:
            type = 'avg'
        layer['pp_' + str(layer_c)] = {'k': 2, 's': 2, 'p': 0, 'type':type}
        layer['mc_' + str(layer_c+1)] = mc

    return gen_succ, layer

#fc layer
def gen_fc(map, layer_f, stop):
    layer = {}
    channel = [1024, 2048, 4096]
    layer['pf_'+str(layer_f)] = {'cin':map['c'], 'cout':channel[random.randint(0,2)]}
    if stop == True:
        layer['pf_' + str(layer_f)]['cout'] = 1000

    layer['mf_'+str(layer_f+1)] = {'c':layer['pf_'+str(layer_f)]['cout']}

    return layer

def gen_VGG():
    model = {}
    model['class'] = 'vgg'
    model['bn']    = False
    model['bias']  = False
    if get_random() >= 0.5:
        model['bn']   = True
    if get_random() >= 0.5:
        model['bias'] = True

    model['net']   = {}

    model['net']['mc_1'] = {'mw':224, 'mh':224, 'c':3}
    stop    = False
    layer_c = 1

    while stop == False:
        rand_seed = get_random()
        if model['net']['mc_' + str(layer_c)]['mw'] < 7 or model['net']['mc_' + str(layer_c)]['mh'] < 7:

            if rand_seed <= 0.7:#go to fc layer
                add_layer = False
                stop      = True

            elif rand_seed > 0.7 and rand_seed <= 0.9:#go to conv layer
                add_layer, layer = gen_conv(model['net']['mc_' + str(layer_c)],layer_c)

            else:#go to pooling layer
                add_layer, layer = gen_pool(model['net']['mc_' + str(layer_c)], layer_c)

        else:
            if rand_seed <= 0.8:#go to conv layer
                add_layer, layer = gen_conv(model['net']['mc_' + str(layer_c)], layer_c)

            else:  # go to pooling layer
                add_layer, layer = gen_pool(model['net']['mc_' + str(layer_c)], layer_c)


        if add_layer == True:
            layer_c += 1
            for e in layer:
                model['net'][e] = layer[e]

    model['net']['mf_1']      = {}
    model['net']['mf_1']['c'] = model['net']['mc_'+str(layer_c)]['mw'] * model['net']['mc_'+str(layer_c)]['mh'] *\
                                model['net']['mc_'+str(layer_c)]['c']
    stop    = False
    layer_f = 1

    while stop == False:
        rand_seed = get_random()
        if rand_seed >= 0.7:
            stop = True

        layer = gen_fc(model['net']['mf_' + str(layer_f)], layer_f, stop)
        for e in layer:
            model['net'][e] = layer[e]

        layer_f += 1

    model['# of layers'] = [layer_c, layer_f]
    return model

if __name__ == '__main__':
    main()
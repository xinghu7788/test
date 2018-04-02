import sys

def read_vgg(i):
    f = open('./graph/VGG'+i+'.txt', 'r')
    a = f.read()
    VGG = eval(a)
    f.close()


    print('--------' + str(i) + '--------')
    print('model type: ' + VGG['class'])

    # if contain BN
    if_bn = VGG['bn']
    print('contain bn: ' + str(if_bn))

    #if contain bias
    if_bias = VGG['bias']
    print('contain bias: ' + str(if_bias))

    # get # of layers for conv and fc
    layer_c = VGG['# of layers'][0]
    layer_f = VGG['# of layers'][1]

    #conv
    c = 1
    while c < layer_c:
        #feature map size
        mc = 'mc_'+str(c)
        print(mc+': '+str(VGG['net'][mc]))

        #conv
        pc = 'pc_'+str(c)
        if pc in VGG['net']:
            print(pc+': '+str(VGG['net'][pc]))

        #pool
        pp = 'pp_'+str(c)
        if pp in VGG['net']:
            print(pp+': '+str(VGG['net'][pp]))

        c += 1

    #last conv layer
    mc = 'mc_' + str(layer_c)
    print(mc + ': ' + str(VGG['net'][mc]))

    f = 1
    while f < layer_f:
        #feature map size
        mf = 'mf_'+str(f)
        print(mf+': '+str(VGG['net'][mf]))
        #parameter
        pf = 'pf_'+str(f)
        print(pf+': '+str(VGG['net'][pf]))

        f += 1

    #last fc layer
    mf = 'mf_' + str(layer_f)
    print(mf + ': ' + str(VGG['net'][mf]))

def main():
    read_vgg(sys.argv[1])

if __name__ == '__main__':
    main()
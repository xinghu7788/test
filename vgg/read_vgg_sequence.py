'''
VGG label
{'label':[], 'para':[]}
conv para: [kw, kh, sw, sh, pw, ph, cin, cout]
pool para: [k, s, p, type]
fc   para: [cin, cout]

label: conv, bn, relu, pooling, fc
'''

#open file VGG.txt
import argparse
import os

def open_file(i):
    f = open('./graph/VGG'+str(i)+'.txt' , 'r')
    a = f.read()
    VGG = eval(a)
    f.close()
    return VGG

def read_sequence(model):
    sequence = {'label':[], 'para':[]}
    if_bn = model['bn']
    layer_c = model['# of layers'][0]
    layer_f = model['# of layers'][1]

    #conv
    c = 1
    while c < layer_c:
        #conv
        pc = 'pc_'+str(c)
        if pc in model['net']:
            kw = model['net'][pc]['kw']
            kh = model['net'][pc]['kh']
            sw = model['net'][pc]['sw']
            sh = model['net'][pc]['sh']
            pw = model['net'][pc]['pw']
            ph = model['net'][pc]['ph']

            cin  = model['net'][pc]['cin']
            cout = model['net'][pc]['cout']

            sequence['label'].append('conv')
            sequence['para'].append([kw, kh, sw, sh, pw, ph, cin, cout])

            if if_bn:
                sequence['label'].append('bn')
                sequence['para'].append([cout])

            sequence['label'].append('relu')
            sequence['para'].append([])

        #pooling
        pp = 'pp_'+str(c)
        if pp in model['net']:
            type = model['net'][pp]['type']

            sequence['label'].append('pooling')
            sequence['para'].append([2, 2, 0, type])

        c += 1

    #fc
    f = 1
    while f < layer_f:
        pf = 'pf_'+str(f)
        cin = model['net'][pf]['cin']
        cout= model['net'][pf]['cout']

        sequence['label'].append('fc')
        sequence['para'].append([cin, cout])
        if f != layer_f - 1:
            sequence['label'].append('relu')
            sequence['para'].append([])


        f += 1

    return sequence

def main():
    #graph = open_file(sys.argv[1])
    #VGG_model = read_sequence(graph)
    #print(VGG_model)
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data')
    args = parser.parse_args()

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    for i in range(3000):
        graph = open_file(str(i))
        VGG_model = read_sequence(graph)
        label_filepath = os.path.join(out_dir, 'vgg_label%d.log' % i)
        with open(label_filepath, 'w') as outfile:
            outfile.write('\n'.join(VGG_model['label']))
            outfile.write('\n')

        para_filepath = os.path.join(out_dir, 'vgg_para%d.log' % i)
        with open(para_filepath, 'w') as outfile:
            for int_list in VGG_model['para']:
                str_list = [str(j) for j in int_list]
                outfile.write('\t'.join(str_list))
                outfile.write('\n')

if __name__ == '__main__':
    main()










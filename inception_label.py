def BasicConv2d(p):
    layer = {'label':[], 'para':[]}
    layer['label'].append('conv')
    layer['para'].append(p)
    layer['label'].append('bn')
    layer['para'].append([p[-1]])
    layer['label'].append('relu')
    layer['para'].append([])

    return layer

def InceptionA(cin, cout):
    layer = {'label':[], 'para':[]}
    # [kw, kh, sw, sh, pw, ph, cin, cout]
    A_list = [[1, 1, 1, 1, 0, 0, cin, 64],
              [1, 1, 1, 1, 0, 0, cin, 48],
              [5, 5, 1, 1, 2, 2, 48,  64],
              [1, 1, 1, 1, 0, 0, cin, 64],
              [3, 3, 1, 1, 1, 1, 64,  96],
              [3, 3, 1, 1, 1, 1, 96,  96]]

    for i in range(len(A_list)):
        temp_layer = BasicConv2d(A_list[i])
        layer['label'] += temp_layer['label']
        layer['para']  += temp_layer['para']

    layer['label'].append('pooling')
    layer['para'].append([3, 1, 1, 'avg'])#[k, s, p, type]

    temp_layer = BasicConv2d([1, 1, 1, 1, 0, 0, cin, cout])
    layer['label'] += temp_layer['label']
    layer['para'] += temp_layer['para']

    layer['label'].append('cat')
    layer['para'].append([])

    return layer

def InceptionB(cin):
    layer = {'label':[], 'para':[]}
    B_list = [[3, 3, 2, 2, 0, 0, cin, 384],
              [1, 1, 1, 1, 0, 0, cin, 64],
              [3, 3, 1, 1, 1, 1, 64,  96],
              [3, 3, 2, 2, 0, 0, 96,  96]]

    for i in range(len(B_list)):
        temp_layer = BasicConv2d(B_list[i])
        layer['label'] += temp_layer['label']
        layer['para']  += temp_layer['para']

    layer['label'].append('pooling')
    layer['para'].append([3, 2, 0, 'max'])  # [k, s, p, type]

    layer['label'].append('cat')
    layer['para'].append([])

    return layer

def InceptionC(cin, cout):
    layer = {'label':[], 'para':[]}
    C_list = [[1, 1, 1, 1, 0, 0, cin,  192],
              [1, 1, 1, 1, 0, 0, cin,  cout],
              [7, 1, 1, 1, 3, 0, cout, cout],
              [1, 7, 1, 1, 0, 3, cout, 192],
              [1, 1, 1, 1, 0, 0, cin,  cout],
              [1, 7, 1, 1, 0, 3, cout, cout],
              [7, 1, 1, 1, 3, 0, cout, cout],
              [1, 7, 1, 1, 0, 3, cout, cout],
              [7, 1, 1, 1, 3, 0, cout, 192]]

    for i in range(len(C_list)):
        temp_layer = BasicConv2d(C_list[i])
        layer['label'] += temp_layer['label']
        layer['para'] += temp_layer['para']

    layer['label'].append('pooling')
    layer['para'].append([3, 1, 1, 'avg'])  # [k, s, p, type]

    temp_layer = BasicConv2d([1, 1, 1, 1, 0, 0, cin, 192])
    layer['label'] += temp_layer['label']
    layer['para'] += temp_layer['para']

    layer['label'].append('cat')
    layer['para'].append([])

    return layer

def InceptionD(cin):
    layer = {'label':[], 'para':[]}
    D_list = [[1, 1, 1, 1, 0, 0, cin, 192],
              [3, 3, 2, 2, 0, 0, 192, 320],
              [1, 1, 1, 1, 0, 0, cin, 192],
              [7, 1, 1, 1, 3, 0, 192, 192],
              [1, 7, 1, 1, 0, 3, 192, 192],
              [3, 3, 2, 2, 0, 0, 192, 192]]

    for i in range(len(D_list)):
        temp_layer = BasicConv2d(D_list[i])
        layer['label'] += temp_layer['label']
        layer['para']  += temp_layer['para']

    layer['label'].append('pooling')
    layer['para'].append([3, 2, 0, 'max'])  # [k, s, p, type]

    layer['label'].append('cat')
    layer['para'].append([])

    return layer

def InceptionE(cin):
    layer = {'label':[], 'para':[]}
    E_list = [[1, 1, 1, 1, 0, 0, cin, 320],
              [1, 1, 1, 1, 0, 0, cin, 384],
              [3, 1, 1, 1, 1, 0, 384, 384],
              [1, 3, 1, 1, 0, 1, 384, 384],
              [1, 1, 1, 1, 0, 0, cin, 448],
              [3, 3, 1, 1, 1, 1, 448, 384],
              [3, 1, 1, 1, 1, 0, 384, 384],
              [1, 3, 1, 1, 0, 1, 384, 384]]

    for i in range(len(E_list)):
        temp_layer = BasicConv2d(E_list[i])
        layer['label'] += temp_layer['label']
        layer['para'] += temp_layer['para']
        if i == 3 or i == 7:
            layer['label'].append('cat')
            layer['para'].append([])

    layer['label'].append('pooling')
    layer['para'].append([3, 1, 1, 'avg'])  # [k, s, p, type]

    temp_layer = BasicConv2d([1, 1, 1, 1, 0, 0, cin, 192])
    layer['label'] += temp_layer['label']
    layer['para'] += temp_layer['para']

    layer['label'].append('cat')
    layer['para'].append([])

    return layer



def model():
    incepion_model = {'label':[], 'para':[]}
    conv_list = [[3, 3, 2, 2, 0, 0, 3,  32],
                 [3, 3, 1, 1, 0, 0, 32, 32],
                 [3, 3, 1, 1, 1, 1, 32, 64],
                 [1, 1, 1, 1, 0, 0, 64, 80],
                 [3, 3, 1, 1, 0, 0, 80, 192]]

    for i in range(len(conv_list)):
        temp_layer = BasicConv2d(conv_list[i])
        incepion_model['label'] += temp_layer['label']
        incepion_model['para']  += temp_layer['para']
        if i == 2 or i == 4:
            incepion_model['label'].append('pooling')
            incepion_model['para'].append([3, 2, 0, 'max'])

    A_list = [[192, 32], [256, 64], [288, 64]]
    for i in range(3):
        temp_layer = InceptionA(A_list[i][0], A_list[i][1])
        incepion_model['label'] += temp_layer['label']
        incepion_model['para']  += temp_layer['para']

    temp_layer = InceptionB(288)
    incepion_model['label'] += temp_layer['label']
    incepion_model['para']  += temp_layer['para']

    C_list = [[768, 128], [768, 160], [768, 160], [768, 192]]
    for i in range(4):
        temp_layer = InceptionC(C_list[i][0], C_list[i][1])
        incepion_model['label'] += temp_layer['label']
        incepion_model['para']  += temp_layer['para']

    temp_layer = InceptionD(768)
    incepion_model['label'] += temp_layer['label']
    incepion_model['para']  += temp_layer['para']

    E_list = [1280, 2048]
    for i in range(2):
        temp_layer = InceptionD(E_list[i])
        incepion_model['label'] += temp_layer['label']
        incepion_model['para']  += temp_layer['para']

    incepion_model['label'].append('pooling')
    incepion_model['para'].append([8, 1, 0, 'avg'])

    incepion_model['label'].append('fc')
    incepion_model['para'].append([2048, 1000])

    return incepion_model

def main():
    inception_model = model()
    print(inception_model)

if __name__ == '__main__':
    main()
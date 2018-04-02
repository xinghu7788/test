import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import sys

def wv(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))

def bv(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(name, inputs, filters, b, stride, if_bn, if_bias):
    if if_bn:
        if if_bias:
            return tf.nn.relu(slim.batch_norm(tf.nn.bias_add(tf.nn.conv2d(inputs, filters, strides=stride, padding='SAME'), b), is_training=True), name=name)
        else:
            return tf.nn.relu(slim.batch_norm(tf.nn.conv2d(inputs, filters, strides=stride, padding='SAME'), is_training=True), name=name)
    else:
        if if_bias:
            return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(inputs, filters, strides=stride, padding='SAME'), b), name=name)
        else:
            return tf.nn.relu(tf.nn.conv2d(inputs, filters, strides=stride, padding='SAME'), name=name)

def max_pool(name,inputs):
    return tf.nn.max_pool(inputs,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name=name)

def avg_pool(name,inputs):
    return tf.nn.avg_pool(inputs,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name=name)

def get_para(para_dic):
    kw = para_dic['kw']
    sw = para_dic['sw']
    pw = para_dic['pw']
    kh = para_dic['kh']
    sh = para_dic['sh']
    ph = para_dic['ph']

    cin = para_dic['cin']
    cout = para_dic['cout']

    return [cin, cout, kh, kw, sh, sw, ph, pw]


def VGG(cfg):
    layer_c = cfg['# of layers'][0]
    layer_f = cfg['# of layers'][1]
    if_bn   = cfg['bn']
    if_bias = cfg['bias']

    g = tf.Graph()
    with g.as_default():
        x = tf.random_normal(shape=[1, 224, 224, 3], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
        x = tf.nn.softplus(x)

        #conv and pooling
        c = 1
        while c < layer_c:
            pc = 'pc_'+str(c)
            pp = 'pp_'+str(c)

            #conv
            if pc in cfg['net']:
                p = get_para(cfg['net'][pc])
                x = conv2d(pc, x, wv([p[2], p[3], p[0], p[1]]), bv([p[1]]), [1, p[4], p[5], 1], if_bn, if_bias)

            #pooling
            if pp in cfg['net']:
                if cfg['net'][pp]['type'] == 'max':
                    x = max_pool(pp, x)
                elif cfg['net'][pp]['type'] == 'avg':
                    x = avg_pool(pp, x)

            c += 1

        dense = tf.reshape(x, [-1, cfg['net']['pf_1']['cin']])

        #fc layer
        f = 1
        while f < layer_f:
            pf = 'pf_'+str(f)
            if f != layer_f - 1:
                dense = tf.nn.relu(tf.nn.bias_add(tf.matmul(dense, wv([cfg['net'][pf]['cin'], cfg['net'][pf]['cout']])), bv([cfg['net'][pf]['cout']])), name=pf)
            else:
                dense = tf.nn.bias_add(tf.matmul(dense, wv([cfg['net'][pf]['cin'], cfg['net'][pf]['cout']])), bv([cfg['net'][pf]['cout']]))

            f += 1

        dense = tf.nn.softplus(dense)

        init = tf.global_variables_initializer()

    with tf.Session(graph=g) as sess:
        sess.run(init)
        sess.run(dense)


def main():
    f = open('./graph/VGG' + sys.argv[1] + '.txt', 'r')
    a = f.read()
    cfg = eval(a)
    f.close()

    VGG(cfg)

    print('finish inference: ' + sys.argv[1])

    return 0


if __name__ == '__main__':
    main()


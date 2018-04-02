import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import sys

def wv(shape):
    return np.ones(shape).astype(np.float32)

def bv(shape):
    return np.ones(shape).astype(np.float32)


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
    if_bn = cfg['bn']
    if_bias = cfg['bias']

    G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
    builder = trt.infer.create_infer_builder(G_LOGGER)

    network = builder.create_network()

    data = network.add_input("data", trt.infer.DataType.FLOAT, (3, 224, 224))
    assert (data)

    #add sigmoid
    x = network.add_activation(data, trt.infer.ActivationType.SIGMOID)
    assert(x)

    #conv and pooling
    c = 1
    while c < layer_c:
        pc = 'pc_' + str(c)
        pp = 'pp_' + str(c)

        #conv
        if pc in cfg['net']:
            p = get_para(cfg['net'][pc])
            x = network.add_convolution(x.get_output(0), p[1], (p[2],p[3]),  wv(p[0]*p[1]*p[2]*p[3]), bv(p[1]))
            assert(x)
            x.set_stride((p[4],p[5]))
            x.set_padding((p[6],p[7]))

            x = network.add_activation(x.get_output(0), trt.infer.ActivationType.RELU)
            assert(x)

        # pooling
        if pp in cfg['net']:
            if cfg['net'][pp]['type'] == 'max':
                x = network.add_pooling(x.get_output(0), trt.infer.PoolingType.MAX, (2, 2))
            elif cfg['net'][pp]['type'] == 'avg':
                x = network.add_pooling(x.get_output(0), trt.infer.PoolingType.MAX, (2, 2))
            assert(x)
            x.set_stride((2, 2))

        c += 1

    #fc layer
    f = 1
    while f < layer_f:
        pf = 'pf_'+str(f)
        x = network.add_fully_connected(x.get_output(0), cfg['net'][pf]['cout'], wv(cfg['net'][pf]['cin'] * cfg['net'][pf]['cout']), bv(cfg['net'][pf]['cout']))
        assert(x)
        if f != layer_f - 1:
            x = network.add_activation(x.get_output(0), trt.infer.ActivationType.RELU)
            assert (x)

        f += 1

    x.get_output(0).set_name("prob")
    network.mark_output(x.get_output(0))

    builder.set_max_batch_size(1)
    builder.set_max_workspace_size(1 << 20)

    engine = builder.build_cuda_engine(network)
    network.destroy()
    builder.destroy()

    context = engine.create_execution_context()
    output = np.empty(1000, dtype=np.float32)
    img = np.ones((3,224,224), dtype=np.float32)
    d_input = cuda.mem_alloc(1 * img.size * img.dtype.itemsize)
    d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)
    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()
    # transfer input data to device
    cuda.memcpy_htod_async(d_input, img, stream)
    # execute model
    context.enqueue(1, bindings, stream.handle, None)
    # transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # syncronize threads
    stream.synchronize()

    context.destroy()
    engine.destroy()


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


import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import sys

def wv(shape):
    return np.ones(shape).astype(np.float32)

def bv(shape):
    return np.ones(shape).astype(np.float32)

def VGG(cfg):
    G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
    builder = trt.infer.create_infer_builder(G_LOGGER)
    network = builder.create_network()

    data = network.add_input("data", trt.infer.DataType.FLOAT, (3, 224, 224))
    assert (data)

    # add sigmoid
    x = network.add_activation(data, trt.infer.ActivationType.SIGMOID)
    assert (x)

    #conv and pooling
    cin = 3
    for v in cfg:
        #pooling
        if v == 'M':
            x = network.add_pooling(x.get_output(0), trt.infer.PoolingType.MAX, (2, 2))
            assert (x)
            x.set_stride((2, 2))

        #conv
        else:
            x = network.add_convolution(x.get_output(0), v, (3, 3), wv(3*3*cin*v), bv(v))
            assert (x)
            x.set_stride((1,1))
            x.set_padding((1,1))

            x = network.add_activation(x.get_output(0), trt.infer.ActivationType.RELU)
            assert (x)

            cin = v

    #fc
    x = network.add_fully_connected(x.get_output(0), 512, wv(512 * 7 * 7 * 512), bv(512))
    assert(x)
    x = network.add_activation(x.get_output(0), trt.infer.ActivationType.RELU)
    assert (x)
    x = network.add_fully_connected(x.get_output(0), 512, wv(512 * 512), bv(512))
    assert (x)
    x = network.add_activation(x.get_output(0), trt.infer.ActivationType.RELU)
    assert (x)
    x = network.add_fully_connected(x.get_output(0), 1000, wv(512 * 1000), bv(1000))
    assert (x)

    x.get_output(0).set_name("prob")
    network.mark_output(x.get_output(0))

    builder.set_max_batch_size(1)
    builder.set_max_workspace_size(1<<20)

    engine = builder.build_cuda_engine(network)
    network.destroy()
    builder.destroy()


    context = engine.create_execution_context()
    output = np.empty(1000, dtype=np.float32)
    img = np.ones((3, 224, 224), dtype=np.float32)
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

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg_list(vgg_type):
    if vgg_type == 'vgg11':
        VGG(cfg['A'])
    elif vgg_type == 'vgg13':
        VGG(cfg['B'])
    elif vgg_type == 'vgg16':
        VGG(cfg['D'])
    elif vgg_type == 'vgg19':
        VGG(cfg['E'])

    else:
        print('wrong model name')


def main():

    vgg_list(sys.argv[1])

    print('finish inference: ' + sys.argv[1])

    return 0


if __name__ == '__main__':
    main()
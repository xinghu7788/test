import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import sys

def wv(shape):
    return np.ones(shape).astype(np.float32)

def bv(shape):
    return np.ones(shape).astype(np.float32)

def conv2d(network, x, in_planes, out_planes, stride=1, kernel=3):
    x = network.add_convolution(x.get_output(0), out_planes, (kernel, kernel), wv(in_planes * out_planes * kernel * kernel), bv(out_planes))
    assert (x)
    x.set_stride((stride, stride))
    if kernel == 3:
        x.set_padding((1, 1))

    return x

def conv2d_relu(network, x):
    x = network.add_activation(x.get_output(0), trt.infer.ActivationType.RELU)
    assert (x)

    return x

def BasicBlock(network, xi, inplanes, planes, stride=1, downsample=False):
    #1st conv
    x = conv2d(network, xi, inplanes, planes, stride)
    x = conv2d_relu(network, x)

    #2nd conv
    x = conv2d(network, x, planes, planes)

    #shotcut
    if downsample:
        xi = conv2d(network, xi, inplanes, planes, stride, 1)

    #add
    x = network.add_element_wise(x.get_output(0), xi.get_output(0), 0)
    assert (x)
    x = conv2d_relu(network, x)

    return x


def Bottleneck(network, xi, inplanes, planes, stride=1, downsample=False):
    # 1st conv
    x = conv2d(network, xi, inplanes, planes, kernel=1)
    x = conv2d_relu(network, x)

    # 2nd conv
    x = conv2d(network, x, planes, planes, stride)
    x = conv2d_relu(network, x)

    #3rd con
    x = conv2d(network, x, planes, planes * 4, kernel=1)

    # shotcut
    if downsample:
        xi = conv2d(network, xi, inplanes, planes * 4, stride, 1)

    # add
    x = network.add_element_wise(x.get_output(0), xi.get_output(0), 0)
    assert (x)
    x = conv2d_relu(network, x)

    return x



def make_layer(network, x, expansion, inplanes, planes, blocks, stride=1):
    downsample = False
    if stride !=1 or inplanes != planes * expansion:
        downsample = True

    if expansion == 1:
        x = BasicBlock(network, x, inplanes, planes, stride, downsample)
        inplanes = planes * expansion
        for i in range(1, blocks):
            x = BasicBlock(network, x, inplanes, planes)

    if expansion == 4:
        x = Bottleneck(network, x, inplanes, planes, stride, downsample)
        inplanes = planes * expansion
        for i in range(1, blocks):
            x = Bottleneck(network, x, inplanes, planes)

    return x, inplanes



def ResNet(block, layers):
    G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
    builder = trt.infer.create_infer_builder(G_LOGGER)
    network = builder.create_network()

    data = network.add_input("data", trt.infer.DataType.FLOAT, (3, 224, 224))
    assert (data)

    # add sigmoid
    x = network.add_activation(data, trt.infer.ActivationType.SIGMOID)
    assert (x)

    #first conv
    x = network.add_convolution(x.get_output(0), 64, (7,7), wv(3 * 64 * 7 * 7), bv(64))
    assert (x)
    x.set_stride((2, 2))
    x.set_padding((3, 3))
    #pool
    x = network.add_pooling(x.get_output(0), trt.infer.PoolingType.MAX, (3, 3))
    assert (x)
    x.set_stride((2, 2))
    x.set_padding((1, 1))

    layer1, inplanes = make_layer(network, x, block, 64, 64, layers[0])
    layer2, inplanes = make_layer(network, layer1, block, inplanes, 128, layers[1], 2)
    layer3, inplanes = make_layer(network, layer2, block, inplanes, 256, layers[2], 2)
    layer4, inplanes = make_layer(network, layer3, block, inplanes, 512, layers[3], 2)

    #pool
    x = network.add_pooling(layer4.get_output(0), trt.infer.PoolingType.MAX, (7, 7))
    assert (x)
    x.set_stride((1, 1))
    #fc
    x = network.add_fully_connected(x.get_output(0), 1000, wv(inplanes * 1000), bv(1000))
    assert (x)

    x.get_output(0).set_name("prob")
    network.mark_output(x.get_output(0))

    builder.set_max_batch_size(1)
    builder.set_max_workspace_size(1 << 20)

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


def resnet_list(resnet_type):
    if resnet_type == 'resnet18':
        ResNet(1, [2, 2, 2, 2])

    elif resnet_type == 'resnet34':
        ResNet(1, [3, 4, 6, 3])

    elif resnet_type == 'resnet50':
        ResNet(4, [3, 4, 6, 3])

    elif resnet_type == 'resnet101':
        ResNet(4, [3, 4, 23, 3])

    elif resnet_type == 'resnet152':
        ResNet(4, [3, 8, 36, 3])

def main():
    resnet_list(sys.argv[1])

    print('finish inference: ' + sys.argv[1])

    return 0


if __name__ == '__main__':
    main()
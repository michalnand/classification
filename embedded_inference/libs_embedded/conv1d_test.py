import torch
import numpy 

def conv1d_ref(x, weights, bias, kernel_size, stride, padding):

    output_channels  = weights.shape[0]
    in_channels      = weights.shape[1]
    kernel_size      = weights.shape[2]
    
    layer = torch.nn.Conv1d(in_channels, output_channels, kernel_size, stride, padding)

    layer.weight.data = torch.from_numpy(weights)
    layer.bias.data   = torch.from_numpy(bias)

    xt = torch.from_numpy(x).unsqueeze(0)
    yt = layer(xt)
    y  = yt.squeeze(0).detach().numpy()


    return y


def conv1d_test(x, weights, bias, kernel_size, stride, padding):

    output_channels  = weights.shape[0]
    in_channels      = weights.shape[1]
    kernel_size      = weights.shape[2]

    width            = x.shape[1]

    width_  = (width + 2*padding - (kernel_size - 1) - 1)//stride + 1
    y       = numpy.zeros((output_channels, width_))
    
    for k in range(output_channels):
        for i in range(width_):
            sum = bias[k] 
            for ch in range(in_channels): 
                for ki in range(kernel_size):
                    idx = i*stride + ki
                    sum+= weights[k][ch][ki]*x[ch][idx]

            y[k][i] = sum

    return y



def conv1d(kernels_count, kernel_size, stride, padding, in_channels, width):

    x       = numpy.random.randn(in_channels, width)
    weights = numpy.random.randn(kernels_count, in_channels, kernel_size)
    bias    = numpy.random.randn(kernels_count)


    y_ref   = conv1d_ref(x, weights, bias, kernel_size, stride, padding)
    y_test  = conv1d_test(x, weights, bias, kernel_size, stride, padding)

    #print(x.shape, y_ref.shape, y_test.shape)

    dif     = ((y_ref - y_test)**2).sum()

    if (dif > 0.000001):
        print("Test FAILED\n")
        print("kernels_count    = ", kernels_count)
        print("kernel_size      = ", kernel_size)
        print("stride           = ", stride)
        print("padding          = ", padding)
        print("in_channels      = ", in_channels)
        print("width            = ", width)
        print("\n\n\n\n")

        print(y_ref)
        print(y_test)
        print("\n\n\n\n")
        exit()
        return False

    return True


input_sizes  = [17, 20, 29, 32, 31, 50, 63, 64, 100, 101, 128, 256, 512]
kernel_sizes = [1, 2, 3, 4, 5, 6, 7, 8]
strides      = [1, 2, 3, 4, 5, 6, 7, 8]


kernels_count = 11
in_channels   = 7

for kernel_size in kernel_sizes:
    for input_size in input_sizes:
        for stride in strides:
            conv1d(kernels_count, kernel_size, stride, 0, in_channels, input_size)



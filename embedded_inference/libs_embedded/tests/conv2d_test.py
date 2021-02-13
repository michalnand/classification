import torch
import numpy 

def conv2d_ref(x, weights, bias, kernel_size, stride, padding):

    output_channels  = weights.shape[0]
    in_channels      = weights.shape[1]
    kernel_size      = weights.shape[2]
    
    layer = torch.nn.Conv2d(in_channels, output_channels, kernel_size, stride, padding)

    layer.weight.data = torch.from_numpy(weights)
    layer.bias.data   = torch.from_numpy(bias)

    xt = torch.from_numpy(x).unsqueeze(0)
    yt = layer(xt)
    y  = yt.squeeze(0).detach().numpy()


    return y


def conv2d_test(x, weights, bias, kernel_size, stride, padding):

    output_channels  = weights.shape[0]
    in_channels      = weights.shape[1]
    kernel_size      = weights.shape[2]

    height            = x.shape[1]
    width             = x.shape[2]

    height_  = (height + 2*padding - (kernel_size - 1) - 1)//stride + 1
    width_   = (width  + 2*padding - (kernel_size - 1) - 1)//stride + 1

    y        = numpy.zeros((output_channels, height_, width_))

    for k in range(output_channels):
        for j in range(height_):
            for i in range(width_):

                sum = bias[k]

                for ch in range(in_channels):
                    for kj in range(kernel_size):
                        for ki in range(kernel_size):
                            idx_j = j*stride + kj
                            idx_i = i*stride + ki
                            sum+= weights[k][ch][kj][ki]*x[ch][idx_j][idx_i]

                y[k][j][i] = sum

    return y



def conv2d(kernels_count, kernel_size, stride, padding, in_channels, height, width):

    x       = numpy.random.randn(in_channels, height, width)
    weights = numpy.random.randn(kernels_count, in_channels, kernel_size, kernel_size)
    bias    = numpy.random.randn(kernels_count)
 
    y_ref   = conv2d_ref(x, weights, bias, kernel_size, stride, padding)
    y_test  = conv2d_test(x, weights, bias, kernel_size, stride, padding)

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


input_heights = [17, 19, 20, 21, 30, 31, 32, 34]
input_widths  = [17, 19, 20, 21, 30, 31, 32, 34]
kernel_sizes  = [1, 2, 3, 4, 5]
strides       = [1, 2, 3, 4, 5]


kernels_count = 3
in_channels   = 5

for kernel_size in kernel_sizes:
    for input_height in input_heights:
        for input_width in input_widths:
            for stride in strides:
                conv2d(kernels_count, kernel_size, stride, 0, in_channels, input_height, input_width)



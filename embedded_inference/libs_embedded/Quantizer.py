import numpy

def Quantizer(weights, bias, range_max = 127):
    tmp             = numpy.concatenate([weights.flatten(), bias.flatten()])
    scale           = numpy.max(numpy.abs(tmp))
    
    '''
    print("layer stats")
    print("mean = ", tmp.mean())
    print("std =  ", tmp.std())
    print("max =  ", tmp.max())
    print("min =  ", tmp.min())
    print("scale =", scale)
    print("\n")
    '''

    weights_scaled  = weights/scale
    bias_scaled     = bias/scale
            
    weights_quant   = numpy.clip(range_max*weights_scaled,    -range_max, range_max)
    bias_quant      = numpy.clip(range_max*bias_scaled,       -range_max, range_max)
    

    return weights_quant, bias_quant, int(scale*1024)
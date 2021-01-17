#ifndef _LINEAR_H_
#define _LINEAR_H_

#include <stdint.h>
#include "dot_microkernel.h"


template<   unsigned int in_features, unsigned int out_features, 
            class IO_t, class WEIGHT_t, class ACC_t, int io_max, int weight_max>
void Linear(IO_t *output_buffer, IO_t *input_buffer, const WEIGHT_t *weights, const WEIGHT_t *bias, int scale)
{ 
    for (unsigned int j = 0; j < out_features; j++)
    {
        ACC_t result = dot_microkernel<in_features, IO_t, WEIGHT_t, ACC_t>(input_buffer, weights + j*in_features);
 
        result = ((result + bias[j])*scale)/(1024*weight_max);

        if (io_max != 1) 
        {
            if (result > io_max) 
                result = io_max;
                    
            if (result < -io_max)
                result = -io_max;
        }  

        output_buffer[j] = result;
    }
}


#endif
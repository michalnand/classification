#ifndef _RELU_H_
#define _RELU_H_

#include <stdint.h>

template<class DType>
void ReLU(  DType *output_buffer, 
            DType *input_buffer, 
            unsigned int size)
{
    for (unsigned int i = 0; i < size; i++)
    {
        if (input_buffer[i] > 0)
            output_buffer[i] = input_buffer[i];
        else
            output_buffer[i] = 0;
    } 
}

#endif
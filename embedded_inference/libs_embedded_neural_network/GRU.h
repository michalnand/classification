#ifndef _GRU_H_
#define _GRU_H_

#include <stdint.h>
#include <typeinfo>

#include "dot_microkernel.h"


template<class IO_t, class ACC_t, int io_max, int scale>
ACC_t _quantize(ACC_t r)
{
    if (typeid(IO_t) == typeid(float))
    {
        result = (r*scale)/1024;
    }
    else 
    {
        result = (r*scale)/(128*1024);
    }

    if (io_max != 0) 
    {
        if (result > io_max) 
            result = io_max;
                    
        if (result < -io_max)
            result = -io_max;
    }

    return result;  
}

//TODO
template<class DType>
DType _tanh(DType x)
{
    return x;
}

//TODO
template<class DType>
DType _sigmoid(DType x)
{
    return x;
}

template<   unsigned int in_features, unsigned int hidden_size, 
            class IO_t, class WEIGHT_t, class ACC_t, int io_max, int scale>
void GRUStep(   IO_t *output_buffer, IO_t *input_buffer, IO_t *hidden,
            
                const WEIGHT_t *wir, const WEIGHT_t *wiz, const WEIGHT_t *win, 
                const WEIGHT_t *whr, const WEIGHT_t *whz, const WEIGHT_t *whn, 
                const WEIGHT_t *bir, const WEIGHT_t *biz, const WEIGHT_t *bin, 
                const WEIGHT_t *bhr, const WEIGHT_t *bhz, const WEIGHT_t *bhn )
{
    for (unsigned int j = 0; j < hidden_size; j++)
    {
        ACC_t r = 0;
        r+= dot_microkernel<in_features, IO_t, WEIGHT_t, ACC_t>(wir + j*in_features, input_buffer) + bir[j];
        r+= dot_microkernel<hidden_size, IO_t, WEIGHT_t, ACC_t>(whr + j*hidden_size, hidden) + bhr[j];
        r = _quantize<IO_t, ACC_t, io_max, scale>(r);
        r = _sigmoid<IO_t>(r);

        ACC_t z = 0;
        z+= dot_microkernel<in_features, IO_t, WEIGHT_t, ACC_t>(wiz + j*in_features, input_buffer) + biz[j];
        z+= dot_microkernel<hidden_size, IO_t, WEIGHT_t, ACC_t>(whz + j*hidden_size, hidden) + bhz[j];
        z = _quantize<IO_t, ACC_t, io_max, scale>(z);
        z = _sigmoid<IO_t>(z);
        
        ACC_t n = 0;
        n+= dot_microkernel<in_features, IO_t, WEIGHT_t, ACC_t>(win + j*in_features, input_buffer) + bin[j];
        ACC_t tmp = r*(dot_microkernel<hidden_size, IO_t, WEIGHT_t, ACC_t>(whn + j*hidden_size, hidden) + bhn[j]);
        
        if (typeid(IO_t) != typeid(float))
            tmp = tmp/127;
        
        n+= tmp;
        
        n = _quantize<IO_t, ACC_t, io_max, scale>(n);
        n = _tanh<IO_t>(n);


        if (typeid(IO_t) == typeid(float))
        {
            output_buffer[j] = (1.0 - z)*n + z*hidden[j];
        }
        else 
        {
            output_buffer[j] = ((127 - z)*n + z*hidden[j])/127;
        }
    }
}


template<   unsigned int in_features, unsigned int hidden_size, 
            class IO_t, class WEIGHT_t, class ACC_t, int io_max, int scale>
void GRU(   IO_t *output_buffer, IO_t *input_buffer, unsigned int sequence_length,
            
            const WEIGHT_t *wir, const WEIGHT_t *wiz, const WEIGHT_t *win, 
            const WEIGHT_t *whr, const WEIGHT_t *whz, const WEIGHT_t *whn, 
            const WEIGHT_t *bir, const WEIGHT_t *biz, const WEIGHT_t *bin, 
            const WEIGHT_t *bhr, const WEIGHT_t *bhz, const WEIGHT_t *bhn )
{ 
    auto hidden_next = output_buffer;
    auto hidden      = output_buffer;

    for (unsigned int j = 0; j < hidden_size; j++)
        hidden[j] = 0;
        
    for (unsigned int t = 0; t < sequence_length; t++)
    {    
        GRUStep<in_features, hidden_size, IO_t, WEIGHT_t, ACC_t, io_max, scale>(
            hidden_next, input_buffer, hidden, 
            wir, wiz, win,
            whr, whz, whn,
            bir, biz, bin,
            bhr, bhz, bhn );
    }
}


#endif
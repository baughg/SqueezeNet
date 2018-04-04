#ifndef OPERATION_H
#define OPERATION_H
#include <stdint.h>

typedef enum 
{
  OP_NONE,
  OP_CONV
}operation_type;

typedef struct convolution_op
{
  uint32_t type;
  uint32_t input_width;
  uint32_t input_height;
  uint32_t input_channels;
  uint32_t output_channels;
  uint32_t kernel_size;
  uint32_t stride;  
  uint32_t pad;
}convolution_op;

typedef enum 
{
  DIR_OUT,
  DIR_IN
}io_direction;

typedef enum 
{
  DT_WEIGHT = 1,
  DT_BIAS,
  DT_ACTIVATION
}io_data_type;

typedef struct operation_io
{
  uint16_t layer;
  uint16_t item;
  uint16_t direction;
  uint16_t data_type;
}operation_io;


#endif
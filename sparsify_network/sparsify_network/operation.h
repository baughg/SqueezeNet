#ifndef OPERATION_H
#define OPERATION_H
#include <stdint.h>
#include <string>
#include <vector>

typedef enum 
{
  OP_NONE,
  OP_CONV
}operation_type;

typedef enum
{
  Z_MAJOR,
  CHANNEL_MAJOR
}storage_order;

typedef struct operation_header
{
  uint32_t type;
  uint32_t size;
  uint32_t inputs;
  uint32_t outputs;
  uint32_t output_configurations;
};

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
  uint32_t storage_order;
}operation_io;

typedef struct operation_config
{
  uint16_t layer;
  uint16_t item;
  uint32_t scale;
}operation_config;

void generate_operation_list(const std::string &root_dir);

void process_convolution(
  uint32_t &op_id, 
  std::string &op_filename,
  FILE* &op_file);

bool get_io_list(uint32_t &op_id,
  std::vector<operation_io> &io_list);

bool get_io_config_list(uint32_t &op_id,
  std::vector<operation_config> &io_cfg_list);
#endif
#include "operation.h"
#include <stdio.h>
#include <string>

operation_type get_operation_type(std::string op_filename);
std::string global_root_dir;

void generate_operation_list(const std::string &root_dir)
{
  global_root_dir = root_dir;
  uint32_t op = 1;
  char operation_filename_str[256];
  std::string operation_filename;
  FILE* op_file = NULL;
  std::string op_filename = root_dir;
  op_filename.append("squeezenet.op");
  op_file = fopen(op_filename.c_str(), "wb");
  bool done = false;

  while (!done)
  {
    sprintf(operation_filename_str, "op%u.bin", op);
    operation_filename = root_dir;
    operation_filename.append(std::string(operation_filename_str));

    switch (get_operation_type(operation_filename))
    {
    case OP_CONV:
      process_convolution(op, operation_filename, op_file);
      break;
    default:
      done = true;
      printf("Unknown operation!\n");
      break;
    }
    op++;
  }

  fclose(op_file);
}

operation_type get_operation_type(std::string op_filename)
{
  operation_type op = OP_NONE;

  FILE* file = NULL;

  file = fopen(op_filename.c_str(), "rb+");

  if (file)
  {
    uint32_t u_op = 0;
    fread(&u_op, sizeof(u_op), 1, file);
    fclose(file);

    op = (operation_type)u_op;   
  }

  return op;
}

void process_convolution(
  uint32_t &op_id, 
  std::string &op_filename,
  FILE* &op_file)
{
  FILE* file = NULL;

  file = fopen(op_filename.c_str(), "rb+");
  convolution_op conv_op;
  std::vector<operation_io> operation_io_list;
  std::vector<operation_config> io_cfg_list;
  std::vector<operation_io> input_list;
  std::vector<operation_io> output_list;

  operation_header header;
  memset(&header, 0, sizeof(header));
  header.type = (uint32_t)OP_CONV;

  if (file)
  {    
    fread(&conv_op, sizeof(conv_op), 1, file);
    fclose(file);  
    header.size = sizeof(conv_op);
    // load inputs
    if (get_io_list(op_id, operation_io_list))
    {
      size_t io_count = operation_io_list.size();
      input_list.reserve(io_count);
      output_list.reserve(io_count);
      header.size += (io_count * sizeof(operation_io));

      for (size_t i = 0; i < io_count; ++i)
      {
        if (operation_io_list[i].direction == DIR_IN)
          input_list.push_back(operation_io_list[i]);
        else
          output_list.push_back(operation_io_list[i]);
      }

      header.inputs = (uint32_t)input_list.size();
      header.outputs = (uint32_t)output_list.size();
    }

    // load config
    if (get_io_config_list(op_id, io_cfg_list))
    {
      header.output_configurations = (uint32_t)io_cfg_list.size();
      header.size += (header.output_configurations * sizeof(operation_config));
    }

    fwrite(&header, sizeof(header), 1, op_file);
    fwrite(&conv_op, sizeof(conv_op), 1, op_file);

    if (header.inputs)
      fwrite(&input_list[0], sizeof(operation_io), header.inputs, op_file);

    if (header.outputs)
      fwrite(&output_list[0], sizeof(operation_io), header.outputs, op_file);

    if (header.output_configurations)
      fwrite(&io_cfg_list[0], sizeof(operation_config), header.output_configurations, op_file);
  }
}

bool get_io_list(uint32_t &op_id,
  std::vector<operation_io> &io_list)
{
  char operation_filename_str[256];
  std::string operation_filename;
  sprintf(operation_filename_str, "io%u.bin", op_id);
  operation_filename = global_root_dir;
  operation_filename.append(std::string(operation_filename_str));

  FILE* file = NULL;

  file = fopen(operation_filename.c_str(), "rb");
  operation_io io;
  io_list.clear();
  io_list.reserve(4);

  if (file)
  {
    while (fread(&io, sizeof(io), 1, file))
    {
      io_list.push_back(io);
    }
    fclose(file);

    return io_list.size() > 0;
  }

  return false;
}


bool get_io_config_list(uint32_t &op_id,
  std::vector<operation_config> &io_cfg_list)
{
  char operation_filename_str[256];
  std::string operation_filename;
  sprintf(operation_filename_str, "io_cfg%u.bin", op_id);
  operation_filename = global_root_dir;
  operation_filename.append(std::string(operation_filename_str));

  FILE* file = NULL;

  file = fopen(operation_filename.c_str(), "rb");
  operation_config cfg;
  io_cfg_list.clear();
  io_cfg_list.reserve(4);

  if (file)
  {
    while (fread(&cfg, sizeof(cfg), 1, file))
    {
      io_cfg_list.push_back(cfg);
    }
    fclose(file);

    return io_cfg_list.size() > 0;
  }

  return false;
}
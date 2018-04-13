#include "operation.h"
#include "sparsity.h"
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
  fwrite(&op, sizeof(op), 1, op_file);

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

  op -= 2;
  op_file = fopen(op_filename.c_str(), "rb+");
  fseek(op_file, 0, SEEK_SET); 
  fwrite(&op, sizeof(op), 1, op_file);
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
  uint32_t output_width = 0;
  uint32_t output_height = 0;
  uint32_t output_channels = 0;
  std::vector<uint8_t> dummy_data;

  std::vector<uint8_t> packed_data;
  std::vector<uint8_t> sparsity_map;
  std::vector<uint32_t> storage_element_relative_address;
  std::vector<uint32_t> sparse_map_relative_address;
  storage_element_header se_header;
  memset(&se_header, 0, sizeof(se_header));

  if (file)
  {    
    fread(&conv_op, sizeof(conv_op), 1, file);
    fclose(file);  

    output_width = conv_op.input_width - ((conv_op.kernel_size >> 1) << 1);
    output_width /= conv_op.stride;
    output_width += (conv_op.stride >> 1);

    output_height = conv_op.input_height - ((conv_op.kernel_size >> 1) << 1);
    output_height /= conv_op.stride;
    output_height += (conv_op.stride >> 1);

    dummy_data.resize(output_width*output_height*conv_op.output_channels);
    memset(&dummy_data[0], 0xff, dummy_data.size());

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

    if (header.outputs)
    {
      switch (output_list[0].storage_order)
      {
      case CHANNEL_MAJOR:
        se_header.order = CHANNEL_MAJOR;
        se_header.layer = output_list[0].layer;
        se_header.item = output_list[0].item;

        build_storage_elements_XYZ(
          dummy_data,
          output_width,
          output_height,
          conv_op.output_channels,
          packed_data,
          sparsity_map,
          storage_element_relative_address,
          sparse_map_relative_address,
          se_header);
        break;
      case Z_MAJOR:
        se_header.order = Z_MAJOR;
        se_header.layer = output_list[0].layer;
        se_header.item = output_list[0].item;

        build_storage_elements_XYZ(
          dummy_data,
          conv_op.output_channels,
          output_width,
          output_height,
          packed_data,
          sparsity_map,
          storage_element_relative_address,
          sparse_map_relative_address,
          se_header);
        break;
      }

      uint32_t elements = se_header.element_count >> 4;
      elements <<= 4;

      if (elements < se_header.element_count)
        elements += 16;

      uint32_t data_size = elements * se_header.element_aligned_bytes;
      uint32_t sparsity_size = elements * se_header.sparsity_bytes;
      se_header.element_count = elements;
      se_header.sparsity_list_offset = elements << 2;
      se_header.data_offset = se_header.sparsity_list_offset << 1;
      se_header.sparse_map_offset = se_header.data_offset + data_size;
      se_header.size = se_header.sparse_map_offset + sparsity_size;
      header.size += sizeof(se_header);
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

    if (header.outputs)
      fwrite(&se_header, sizeof(se_header), 1, op_file);
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
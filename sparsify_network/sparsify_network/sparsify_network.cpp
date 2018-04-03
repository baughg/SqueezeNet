// sparsify_network.cpp : Defines the entry point for the console application.
// SqueezeNet version

#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>

typedef struct weight_header
{
  uint16_t layer;
  uint16_t item;
  unsigned order;
  unsigned X;
  unsigned Y;
  unsigned Z;
  unsigned weights;
}weight_header;

typedef struct storage_element_header
{
  unsigned size;
  uint16_t layer;
  uint16_t item;
  unsigned order;
  unsigned element_count;
  unsigned element_max_length;
  unsigned element_aligned_bytes;
  unsigned sparsity_bytes;
  unsigned data_offset;
  unsigned sparse_map_offset;
  unsigned data_list_offset;
  unsigned sparsity_list_offset;
}storage_element_header;

void sparsify(
  uint8_t* p_data,
  uint32_t len,
  std::vector<uint8_t> &data,
  std::vector<uint8_t> &sparsity_map);

void build_storage_elements_XYZ(
  std::vector<uint8_t> &data,
  uint32_t X,
  uint32_t Y,
  uint32_t Z,
  std::vector<uint8_t> &packed_data,
  std::vector<uint8_t> &sparsity_map,
  std::vector<uint32_t> &storage_element_relative_address,
  std::vector<uint32_t> &sparse_map_relative_address,
  storage_element_header &se_header);

bool expand(
  int8_t* p_data,
  int8_t* p_sparsity_map,
  uint32_t elements,
  int8_t* p_dense_data);

void write_to_file(std::vector<uint8_t> &data, std::string filename);
void write_to_file_u32(std::vector<uint32_t> &data, std::string filename);
void write_to_file(std::vector<uint8_t> &data, FILE* &file);
void write_to_file_u32(std::vector<uint32_t> &data, FILE* &file);

void densify_weights_zxy(std::string root_dir);

void sparsify_weights_xyz(
  std::string root_dir, 
  std::string output_prefix,
  storage_element_header &se_header,
  FILE* &blob_file);

void sparsify_weights_zxy(
  std::string root_dir,
  std::string output_prefix,
  storage_element_header &se_header,
  FILE* &blob_file);

void sparsify_input_xyz(std::string root_dir);

void sparsify_weights_fc_zxy(std::string root_dir);
unsigned get_order(std::string &filename, std::string &root_dir);

int main(int argc, char** argv)
{
  if (argc <= 1)
  {
    printf("sparsify_network [binary_directory]\n");
    exit(1);
  }

  int mode = 0;

  if (argc >= 3)
  {
    if (strcmp(argv[2], "dense") == 0)
      mode = 1;
    else if (strcmp(argv[2], "zxy") == 0)
      mode = 2;
    else if (strcmp(argv[2], "fc_zxy") == 0)
      mode = 3;
    else if (strcmp(argv[2], "input") == 0)
      mode = 4;
  }

  std::string root_dir(argv[1]);
  

  if (!mode) {
    FILE* blob_file = NULL;
    std::string blob_filename = root_dir;
    blob_filename.append("squeezenet.blob");
    blob_file = fopen(blob_filename.c_str(), "wb");

    std::vector<std::string> file_list;
    file_list.reserve(18);
    file_list.push_back("weight1_0");
    file_list.push_back("bias1_0");
    file_list.push_back("weight2_0");
    file_list.push_back("bias2_0");
    file_list.push_back("weight2_1");
    file_list.push_back("bias2_1");
    file_list.push_back("weight2_2");
    file_list.push_back("bias2_2");
    file_list.push_back("weight3_0");
    file_list.push_back("bias3_0");
    file_list.push_back("weight3_1");
    file_list.push_back("bias3_1");
    file_list.push_back("weight3_2");
    file_list.push_back("bias3_2");
    uint32_t global_offset = 0;
    std::vector<storage_element_header> se_header_list;
    
    const size_t files = file_list.size();
    se_header_list.resize(files);
    const uint32_t header_count = (uint32_t)files;
    fwrite(&header_count, sizeof(uint32_t), 1, blob_file);
    fwrite(&se_header_list[0], sizeof(storage_element_header), files, blob_file);
    global_offset += sizeof(uint32_t);
    global_offset += sizeof(storage_element_header) * files;

    for (size_t f = 0; f < files; ++f) {
      switch (get_order(file_list[f], root_dir)) {
      case 1: // XYZ order
        sparsify_weights_xyz(root_dir, file_list[f], se_header_list[f], blob_file);
        break;
      case 2:
        sparsify_weights_zxy(root_dir, file_list[f], se_header_list[f], blob_file);
        break;
      default:
        printf("Read error!\n");
        break;
      }

      global_offset += sizeof(storage_element_header);
      se_header_list[f].data_offset += global_offset;
      se_header_list[f].data_list_offset += global_offset;
      se_header_list[f].sparse_map_offset += global_offset;
      se_header_list[f].sparsity_list_offset += global_offset;
      global_offset += se_header_list[f].size;
    }

    fclose(blob_file);

    blob_file = fopen(blob_filename.c_str(), "rb+");
    fseek(blob_file, 0, SEEK_SET);
    fwrite(&header_count, sizeof(uint32_t), 1, blob_file);
    fwrite(&se_header_list[0], sizeof(storage_element_header), files, blob_file);
    fclose(blob_file);
  }
  else if(mode == 1)
    densify_weights_zxy(root_dir);  
  else if (mode == 3)
    sparsify_weights_fc_zxy(root_dir);
  if (mode == 4)
    sparsify_input_xyz(root_dir);

  return 0;
}

void densify_weights_zxy(std::string root_dir)
{
  std::vector<uint8_t> buffer;
  std::string output_prefix = "output_c";

  const static uint32_t SE_OUTPUT = 64;
  //const static uint32_t H = 28;
  //const static uint32_t W = 28;
  const static uint32_t H = 28;
  const static uint32_t W = 28;
  const static uint32_t Ho = 1;
  const static uint32_t Wo = 1;
#define OUTPUT_CHANNELS 120

  uint8_t output_data[16][16 * SE_OUTPUT];
  uint8_t output_sparse_map[16][2 * SE_OUTPUT];
  uint32_t output_data_se_address[H*W];
  uint32_t output_sparse_se_address[H*W];
  uint8_t output_data_dense[H*W][OUTPUT_CHANNELS];

  std::string activation_filename = root_dir;
  activation_filename.append(output_prefix + "_packed_data_i8.bin");

  std::string activation_sparsity_filename = root_dir;
  activation_sparsity_filename.append(output_prefix + "_sparsity_map_i8.bin");

  std::string activation_list_filename = root_dir;
  activation_list_filename.append(output_prefix + "_se_data_address_i8.bin");
 
  std::string activation_sparsity_list_filename = root_dir;
  activation_sparsity_list_filename.append(
    output_prefix + "_se_sparsity_address_i8.bin");

  {
    FILE* output_data_file = NULL;
    output_data_file = fopen(activation_filename.c_str(), "rb");
    fread(output_data, 1, sizeof(output_data), output_data_file);
    fclose(output_data_file);

    FILE* output_sparse_map_file = NULL;
    output_sparse_map_file = fopen(activation_sparsity_filename.c_str(), "rb");
    fread(output_sparse_map, 1, sizeof(output_sparse_map), output_sparse_map_file);
    fclose(output_sparse_map_file);

    FILE* addr_file = NULL;
    addr_file = fopen(activation_list_filename.c_str(), "rb");
    fread(output_data_se_address, 1, sizeof(output_data_se_address), addr_file);
    fclose(addr_file);
    
    addr_file = fopen(activation_sparsity_list_filename.c_str(), "rb");
    fread(output_sparse_se_address, 1, sizeof(output_sparse_se_address), addr_file);
    fclose(addr_file);
  }

  const uint32_t storage_elements = Ho*Wo;
  uint32_t min_se_data_addr = output_data_se_address[0];
  uint32_t min_se_sparse_addr = output_sparse_se_address[0];

  for (uint32_t se = 1; se < storage_elements; ++se)
  {
    if (output_data_se_address[se] < min_se_data_addr)
      min_se_data_addr = output_data_se_address[se];

    if (output_sparse_se_address[se] < min_se_sparse_addr)
      min_se_sparse_addr = output_sparse_se_address[se];
  }

  uint8_t* p_se_data = &output_data[0][0];
  uint8_t* p_se_sparse_map = &output_sparse_map[0][0];
  uint32_t data_offset = 0;
  uint32_t sparse_offset = 0;
  int8_t* p_data, *p_sparse_map;

  for (uint32_t se = 0; se < storage_elements; ++se)
  {
    data_offset = output_data_se_address[se] - min_se_data_addr;
    sparse_offset = output_sparse_se_address[se] - min_se_sparse_addr;
    p_data = (int8_t*)p_se_data + data_offset;
    p_sparse_map = (int8_t*)p_se_sparse_map + sparse_offset;

    expand(p_data, p_sparse_map, OUTPUT_CHANNELS, (int8_t*)output_data_dense[se]);
  }

  std::string output_dense_filename = root_dir;
  output_dense_filename.append(
    output_prefix + "_dense_i8.bin");

  FILE* dense_data_file = NULL;
  dense_data_file = fopen(output_dense_filename.c_str(), "wb");
  fwrite(output_data_dense, 1, sizeof(output_data_dense), dense_data_file);
  fclose(dense_data_file);
}

bool expand(
  int8_t* p_data, 
  int8_t* p_sparsity_map, 
  uint32_t elements,
  int8_t* p_dense_data)
{
  if (!p_data || !p_sparsity_map)
    return false;



  int8_t sparse_mask = 0;
  int32_t sparse_byte_index = 0;
  int32_t sparse_bit_index = 0;

  for (int32_t elem = 0; elem < elements; ++elem)
  {
    p_dense_data[elem] = 0;
    sparse_byte_index = elem >> 3;
    sparse_bit_index = elem % 8;
    sparse_mask = p_sparsity_map[sparse_byte_index];
    sparse_mask >>= sparse_bit_index;
    sparse_mask &= 0x1;
    p_dense_data[elem] |= (*p_data * sparse_mask);
    p_data += sparse_mask;
  }

  return true;
}

void sparsify_weights_xyz(
  std::string root_dir, 
  std::string output_prefix,
  storage_element_header &se_header,
  FILE* &blob_file)
{
  std::vector<uint8_t> buffer;

  uint32_t X = 25;
  uint32_t Y = 6;
  
  std::string binary_filename = root_dir;
  
  binary_filename.append(output_prefix + "_i8.bin");
  buffer.resize(X * Y);
  FILE* input_file = NULL;
  input_file = fopen(binary_filename.c_str(), "rb");

  weight_header wght_header;
  unsigned points = 0;

  if (input_file) {
    fread(&wght_header, sizeof(wght_header), 1, input_file);
    X = wght_header.X * wght_header.Y;
    Y = wght_header.Z;
    points = wght_header.X * wght_header.Y * wght_header.Z;
    points *= wght_header.weights;

    buffer.resize(points);
    fread(&buffer[0], 1, buffer.size(), input_file);
    fclose(input_file);
    std::vector<uint8_t> packed_data;
    std::vector<uint8_t> sparsity_map;
    std::vector<uint32_t> storage_element_relative_address;
    std::vector<uint32_t> sparse_map_relative_address;
    
    se_header.layer = wght_header.layer;
    se_header.item = wght_header.item;
    se_header.order = wght_header.order;

    build_storage_elements_XYZ(
      buffer,
      X,
      Y,
      wght_header.weights,
      packed_data,
      sparsity_map,
      storage_element_relative_address,
      sparse_map_relative_address,
      se_header);

    fwrite(&se_header, sizeof(se_header), 1, blob_file);
    write_to_file_u32(storage_element_relative_address, blob_file);
    write_to_file_u32(sparse_map_relative_address, blob_file);
    write_to_file(packed_data, blob_file);
    write_to_file(sparsity_map, blob_file);
  }
}
void sparsify(
  uint8_t* p_data,
  uint32_t len,
  std::vector<uint8_t> &data,
  std::vector<uint8_t> &sparsity_map)
{
  data.clear();
  sparsity_map.clear();
  data.reserve(len);
  uint32_t sparse_mask_bytes = len >> 3;

  if (len > (sparse_mask_bytes << 3))
    sparse_mask_bytes++;


  sparsity_map.resize(sparse_mask_bytes);
  uint32_t byte_index = 0;
  uint32_t bit_index = 0;

  for (uint32_t e = 0; e < len; ++e)
  {
    byte_index = e >> 3;
    bit_index = e % 8;

    if (p_data[e])
    {
      data.push_back(p_data[e]);
      sparsity_map[byte_index] |= (1 << bit_index);
    }
  }
}


void build_storage_elements_XYZ(
  std::vector<uint8_t> &data,
  uint32_t X,
  uint32_t Y,
  uint32_t Z,
  std::vector<uint8_t> &packed_data,
  std::vector<uint8_t> &sparsity_map,
  std::vector<uint32_t> &storage_element_relative_address,
  std::vector<uint32_t> &sparse_map_relative_address,
  storage_element_header &se_header)
{
  std::vector<uint8_t> tensor_packed_data;
  std::vector<uint8_t> tensor_sparsity_map;
  //memset(&se_header, 0, sizeof(se_header));

  const uint32_t tensor_count = Y*Z;
  se_header.element_count = tensor_count;

  uint32_t sparse_mask_bytes = X >> 3;

  if (X > (sparse_mask_bytes << 3))
    sparse_mask_bytes++;

  se_header.sparsity_bytes = sparse_mask_bytes;

  sparse_map_relative_address.resize(tensor_count);
  storage_element_relative_address.resize(tensor_count);
  sparsity_map.resize(tensor_count * sparse_mask_bytes);
  uint8_t* p_data = &data[0];
  uint32_t addr = 0;
  uint32_t sparse_addr = 0;
  uint32_t addr_non_zero = 0;
  uint32_t index = 0;
  uint32_t storage_element_size = X >> 4;

  if ((storage_element_size << 4) < X)
    storage_element_size++;

  storage_element_size <<= 4;
  se_header.element_aligned_bytes = storage_element_size;
  se_header.element_max_length = X;
  packed_data.resize(tensor_count * storage_element_size);

  uint8_t* p_tensor_packed_data = &packed_data[0];
  uint8_t* p_tensor_sparsity_map = &sparsity_map[0];

  for (uint32_t z = 0; z < Z; ++z)
    for (uint32_t y = 0; y < Y; ++y)
    {
      sparsify(p_data, X, tensor_packed_data, tensor_sparsity_map);

      if (tensor_packed_data.size())
      {
        memcpy(p_tensor_packed_data, &tensor_packed_data[0], tensor_packed_data.size());
        p_tensor_packed_data += storage_element_size;
        addr = addr_non_zero;
        addr_non_zero += storage_element_size;
      }
      else
      {
        addr = ~0;
      }

      
      storage_element_relative_address[index] = addr;
      memcpy(p_tensor_sparsity_map, &tensor_sparsity_map[0], sparse_mask_bytes);
      p_tensor_sparsity_map += sparse_mask_bytes;
      sparse_map_relative_address[index++] = sparse_addr;
      sparse_addr += sparse_mask_bytes;
      p_data += X;
    }

  addr_non_zero += storage_element_size;
  packed_data.resize(addr_non_zero);
  addr_non_zero -= storage_element_size;

  for (uint32_t t = 0; t < tensor_count; ++t)
  {
    if (storage_element_relative_address[t] == ~0)
    {
      storage_element_relative_address[t] = addr_non_zero;
    }
  }
  size_t sparsity_map_size = sparsity_map.size();

  // align to word boundary
  sparsity_map_size >>= 2;
  sparsity_map_size <<= 2;

  if (sparsity_map_size < sparsity_map.size())
    sparsity_map_size += 4;

  sparsity_map.resize(sparsity_map_size);
  se_header.data_list_offset = 0;
  se_header.sparsity_list_offset = storage_element_relative_address.size() * sizeof(uint32_t);
  se_header.data_offset = se_header.sparsity_list_offset << 1;
  se_header.sparse_map_offset = se_header.data_offset + (uint32_t)packed_data.size();
  se_header.size = se_header.sparse_map_offset + sparsity_map.size();
}

void write_to_file(std::vector<uint8_t> &data, std::string filename)
{
  FILE* file = NULL;

  file = fopen(filename.c_str(), "wb");

  if (file) {
    fwrite(&data[0], 1, data.size(), file);
    fclose(file);
  }
}

void write_to_file(std::vector<uint8_t> &data, FILE* &file)
{  
  if (file) {
    fwrite(&data[0], 1, data.size(), file);    
  }
}

void write_to_file_u32(std::vector<uint32_t> &data, FILE* &file)
{  
  if (file) {
    fwrite(&data[0], sizeof(uint32_t), data.size(), file);    
  }
}

void write_to_file_u32(std::vector<uint32_t> &data, std::string filename)
{
  FILE* file = NULL;

  file = fopen(filename.c_str(), "wb");

  if (file) {
    fwrite(&data[0], sizeof(uint32_t), data.size(), file);
    fclose(file);
  }
}

void sparsify_weights_zxy(
  std::string root_dir, 
  std::string output_prefix,
  storage_element_header &se_header,
  FILE* &blob_file)
{
  std::vector<uint8_t> buffer;
  std::vector<uint8_t> buffer_zxy;

  uint32_t X = 25;
  uint32_t Y = 6;

  std::string binary_filename = root_dir;
  //std::string output_prefix = "weight5_6";

  binary_filename.append(output_prefix + "_i8.bin");
  buffer.resize(X * Y);
  FILE* input_file = NULL;
  input_file = fopen(binary_filename.c_str(), "rb");

  weight_header wght_header;
  unsigned points = 0;

  if (input_file) {
    fread(&wght_header, sizeof(wght_header), 1, input_file);
    X = wght_header.X * wght_header.Y;
    Y = wght_header.Z;
    points = wght_header.X * wght_header.Y * wght_header.Z;
    points *= wght_header.weights;

    buffer.resize(points);
    fread(&buffer[0], 1, buffer.size(), input_file);
    fclose(input_file);
    std::vector<uint8_t> packed_data;
    std::vector<uint8_t> sparsity_map;
    std::vector<uint32_t> storage_element_relative_address;
    std::vector<uint32_t> sparse_map_relative_address;
    

    buffer_zxy.resize(buffer.size());
    uint8_t* p_buffer = &buffer[0];
    uint8_t* p_weight_set = p_buffer;
    uint8_t* p_buffer_zxy = &buffer_zxy[0];
    uint8_t* p_weight_set_zxy = p_buffer_zxy;
    uint32_t channel_size = wght_header.X * wght_header.Y;
    uint32_t channels = wght_header.weights;
    uint32_t weight_set_size = channel_size * channels;
    uint32_t weight_sets = wght_header.Z;
    uint32_t channel_set_size = channel_size * weight_sets;    
    uint32_t offset = 0;

    
    build_storage_elements_XYZ(
      buffer,
      channels,
      channel_size,
      weight_sets,
      packed_data,
      sparsity_map,
      storage_element_relative_address,
      sparse_map_relative_address,
      se_header);
    
    fwrite(&se_header, sizeof(se_header), 1, blob_file);
    write_to_file(packed_data, blob_file);    
    write_to_file(sparsity_map, blob_file);    
    write_to_file_u32(storage_element_relative_address, blob_file);  
    write_to_file_u32(sparse_map_relative_address, blob_file);
  }
}

void sparsify_weights_fc_zxy(std::string root_dir)
{
  std::vector<uint8_t> buffer;
  std::vector<uint8_t> buffer_zxy;

  uint32_t X = 25;
  uint32_t Y = 6;

  std::string binary_filename = root_dir;
  std::string output_prefix = "weight5_6";

  binary_filename.append(output_prefix + "_i8.bin");
  buffer.resize(X * Y);
  FILE* input_file = NULL;
  input_file = fopen(binary_filename.c_str(), "rb");

  weight_header wght_header;
  unsigned points = 0;

  if (input_file) {
    fread(&wght_header, sizeof(wght_header), 1, input_file);
    X = wght_header.X * wght_header.Y;
    Y = wght_header.Z;
    points = wght_header.X * wght_header.Y * wght_header.Z;
    points *= wght_header.weights;

    buffer.resize(points);
    fread(&buffer[0], 1, buffer.size(), input_file);
    fclose(input_file);
    std::vector<uint8_t> packed_data;
    std::vector<uint8_t> sparsity_map;
    std::vector<uint32_t> storage_element_relative_address;
    std::vector<uint32_t> sparse_map_relative_address;

    buffer_zxy.resize(buffer.size());
    uint8_t* p_buffer = &buffer[0];
    uint8_t* p_weight_set = p_buffer;
    uint8_t* p_buffer_zxy = &buffer_zxy[0];
    uint8_t* p_weight_set_zxy = p_buffer_zxy;
    uint32_t channel_size = wght_header.weights * wght_header.Z;
    uint32_t channels = wght_header.Y;
    uint32_t weight_set_size = channel_size * channels;
    uint32_t weight_sets = wght_header.X;
    uint32_t channel_set_size = channel_size * weight_sets;
    uint32_t offset = 0;

    /*for (uint32_t ws = 0; ws < weight_sets; ++ws)
    {
      p_weight_set_zxy = p_buffer_zxy + ws * weight_set_size;

      for (uint32_t c = 0; c < channels; ++c) {
        p_weight_set = p_buffer + ws * channel_size + c * channel_set_size;

        for (uint32_t e = 0; e < channel_size; ++e)
        {
          offset = e * channels + c;
          p_weight_set_zxy[offset] = p_weight_set[e];
        }

        p_weight_set += channel_size;
      }
    }*/
    storage_element_header se_header;

    build_storage_elements_XYZ(
      buffer,
      channels,
      channel_size,
      weight_sets,
      packed_data,
      sparsity_map,
      storage_element_relative_address,
      sparse_map_relative_address,
      se_header);


    std::string filename = root_dir;
    filename.append(output_prefix + "_packed_data_i8.bin");
    write_to_file(packed_data, filename);
    filename = root_dir;
    filename.append(output_prefix + "_sparsity_map_i8.bin");
    write_to_file(sparsity_map, filename);
    filename = root_dir;
    filename.append(output_prefix + "_se_data_address_i8.bin");
    write_to_file_u32(storage_element_relative_address, filename);
    filename = root_dir;
    filename.append(output_prefix + "_se_sparsity_address_i8.bin");
    write_to_file_u32(sparse_map_relative_address, filename);
  }
}


void sparsify_input_xyz(std::string root_dir)
{
  std::vector<uint8_t> buffer;

  uint32_t X = 28;
  uint32_t Y = 28;

  std::string binary_filename = root_dir;
  std::string output_prefix = "input";

  binary_filename.append(output_prefix + "_i8.bin");
  buffer.resize(X * Y);
  FILE* input_file = NULL;
  input_file = fopen(binary_filename.c_str(), "rb");


  unsigned points = 0;

  if (input_file) {    
    fread(&buffer[0], 1, buffer.size(), input_file);
    fclose(input_file);
    std::vector<uint8_t> packed_data;
    std::vector<uint8_t> sparsity_map;
    std::vector<uint32_t> storage_element_relative_address;
    std::vector<uint32_t> sparse_map_relative_address;
    storage_element_header se_header;

    build_storage_elements_XYZ(
      buffer,
      X,
      Y,
      1,
      packed_data,
      sparsity_map,
      storage_element_relative_address,
      sparse_map_relative_address,
      se_header);


    std::string filename = root_dir;
    filename.append(output_prefix + "_packed_data_i8.bin");
    write_to_file(packed_data, filename);
    filename = root_dir;
    filename.append(output_prefix + "_sparsity_map_i8.bin");
    write_to_file(sparsity_map, filename);
    filename = root_dir;
    filename.append(output_prefix + "_se_data_address_i8.bin");
    write_to_file_u32(storage_element_relative_address, filename);
    filename = root_dir;
    filename.append(output_prefix + "_se_sparsity_address_i8.bin");
    write_to_file_u32(sparse_map_relative_address, filename);
  }
}

unsigned get_order(std::string &filename, std::string &root_dir)
{
  std::string binary_filename = root_dir;

  binary_filename.append(filename + "_i8.bin");

  FILE* input_file = fopen(binary_filename.c_str(), "rb");

  weight_header wght_header;
  unsigned points = 0;

  if (input_file) {
    fread(&wght_header, sizeof(wght_header), 1, input_file);
    fclose(input_file);
    return wght_header.order;
  }

  return ~0;
}
#ifndef SPARSITY_H
#define SPARSITY_H
#include <string>
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

void sparsify_input_xyz(
  std::string root_dir,
  std::string output_prefix,
  storage_element_header &se_header,
  FILE* &blob_file);

void sparsify_weights_fc_zxy(std::string root_dir);
unsigned get_order(std::string &filename, std::string &root_dir);
#endif

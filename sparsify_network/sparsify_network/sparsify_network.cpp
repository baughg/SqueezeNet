// sparsify_network.cpp : Defines the entry point for the console application.
// SqueezeNet version

#include "stdafx.h"
#include "sparsity.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>



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
    if (strcmp(argv[2], "input") == 0)
      mode = 1;    
  }

  std::string root_dir(argv[1]);
  std::vector<std::string> file_list;
  file_list.reserve(18);
  std::string output_filename;
  if (!mode) {
    output_filename = "squeezenet.blob";
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
  }
  else if (mode == 1) {
    output_filename = "input.blob";
    file_list.push_back("input0_0");
  }


  FILE* blob_file = NULL;
  std::string blob_filename = root_dir;
  blob_filename.append(output_filename);
  blob_file = fopen(blob_filename.c_str(), "wb");

  
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
    case 3: // XYZ order input
      sparsify_input_xyz(root_dir, file_list[f], se_header_list[f], blob_file);
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
  return 0;
}


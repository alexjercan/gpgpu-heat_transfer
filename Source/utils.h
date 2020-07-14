#pragma once
#include "CL/cl.h"
#include <d3d9.h>

#define OPENCL_VERSION_1_2  1.2f
#define OPENCL_VERSION_2_0  2.0f

int read_source_from_file(const char* file_name, char** source, size_t* source_size);




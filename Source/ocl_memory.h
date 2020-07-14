#pragma once
#include <CL/cl.h>

struct ocl_args_d_t;

void generate_input(cl_float* input_array, cl_uint array_width, cl_uint array_height, cl_float temperature, cl_uint point_x, cl_uint point_y, cl_float point_temperature);
int create_buffer_arguments(ocl_args_d_t* ocl, cl_float* input, cl_float* output, const cl_uint array_width, const cl_uint array_height);
bool read_and_verify(ocl_args_d_t* ocl, const cl_uint width, const cl_uint height);

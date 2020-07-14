#pragma once
#include <CL/cl.h>

struct ocl_args_d_t;

cl_int setup_ocl_kernel(ocl_args_d_t* ocl, const char* program_name, const char* kernel_name);
cl_uint set_kernel_arguments(ocl_args_d_t* ocl, cl_uint width, cl_uint height, cl_float temperature, char axis);
cl_uint execute_add_kernel(ocl_args_d_t* ocl, const cl_uint width, const cl_uint height);
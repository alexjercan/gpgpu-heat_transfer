#pragma once
#include <CL/cl.h>

struct ocl_args_d_t;

int setup_open_cl(ocl_args_d_t* ocl, const cl_device_type device_type, const char* preferred_platform = "Intel");
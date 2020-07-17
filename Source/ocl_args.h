#pragma once
#include <CL/cl.h>

struct ocl_args_d_t
{
    ocl_args_d_t();
    ~ocl_args_d_t();

    cl_context       context;
    cl_device_id     device;
    cl_command_queue command_queue;
    cl_program       program;
    cl_kernel        kernel;
    float            platform_version;
    float            device_version;
    float            compiler_version;

    cl_mem           input;
    cl_mem           output;
    cl_mem           plate_points;
};



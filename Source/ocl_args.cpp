#include "ocl_args.h"

#include <stdlib.h>
#include <tchar.h>
#include <vector>

#include "CL/cl.h"

#include <Windows.h>


#include "log_utils.h"
#include "utils.h"

ocl_args_d_t::ocl_args_d_t() :
	context(nullptr),
	device(nullptr),
	command_queue(nullptr),
	program(nullptr),
	kernel(nullptr),
	platform_version(OPENCL_VERSION_1_2),
	device_version(OPENCL_VERSION_1_2),
	compiler_version(OPENCL_VERSION_1_2),
	input(nullptr),
	output(nullptr)
{
}

ocl_args_d_t::~ocl_args_d_t()
{
    auto err = CL_SUCCESS;

    if (kernel)
    {
        err = clReleaseKernel(kernel);
        if (CL_SUCCESS != err)
	        log_error("Error: clReleaseKernel returned '%s'.\n", translate_open_cl_error(err));
    }
    if (program)
    {
        err = clReleaseProgram(program);
        if (CL_SUCCESS != err)
	        log_error("Error: clReleaseProgram returned '%s'.\n", translate_open_cl_error(err));
    }
    if (input)
    {
        err = clReleaseMemObject(input);
        if (CL_SUCCESS != err)
	        log_error("Error: clReleaseMemObject returned '%s'.\n", translate_open_cl_error(err));
    }
    if (output)
    {
        err = clReleaseMemObject(output);
        if (CL_SUCCESS != err)
	        log_error("Error: clReleaseMemObject returned '%s'.\n", translate_open_cl_error(err));
    }
    if (command_queue)
    {
        err = clReleaseCommandQueue(command_queue);
        if (CL_SUCCESS != err)
	        log_error("Error: clReleaseCommandQueue returned '%s'.\n", translate_open_cl_error(err));
    }
    if (device)
    {
        err = clReleaseDevice(device);
        if (CL_SUCCESS != err)
	        log_error("Error: clReleaseDevice returned '%s'.\n", translate_open_cl_error(err));
    }
    if (context)
    {
        err = clReleaseContext(context);
        if (CL_SUCCESS != err)
	        log_error("Error: clReleaseContext returned '%s'.\n", translate_open_cl_error(err));
    }
}

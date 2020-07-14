#include "ocl_kernel.h"

#include <vector>


#include "log_utils.h"
#include "ocl_args.h"
#include "utils.h"

static int create_and_build_program(ocl_args_d_t* ocl, const char* program_name);

cl_int setup_ocl_kernel(ocl_args_d_t* ocl, const char* program_name, const char* kernel_name)
{
    cl_int err;

    if (CL_SUCCESS != create_and_build_program(ocl, program_name))
    {
        return -1;
    }

    ocl->kernel = clCreateKernel(ocl->program, kernel_name, &err);
    if (CL_SUCCESS != err)
    {
        log_error("Error: clCreateKernel returned %s\n", translate_open_cl_error(err));
        return -1;
    }

    return CL_SUCCESS;
}

int create_and_build_program(ocl_args_d_t* ocl, const char* program_name)
{
    char* source = nullptr;
    size_t src_size = 0;
    auto err = read_source_from_file(program_name, &source, &src_size);

    if (CL_SUCCESS != err)
    {
        log_error("Error: ReadSourceFromFile returned %s.\n", translate_open_cl_error(err));
        if (source)
        {
            delete[] source;
            source = nullptr;
        }

        return err;
    }

    ocl->program = clCreateProgramWithSource(ocl->context, 1, const_cast<const char**>(&source), &src_size, &err);
    if (CL_SUCCESS != err)
    {
        log_error("Error: clCreateProgramWithSource returned %s.\n", translate_open_cl_error(err));
        if (source)
        {
            delete[] source;
            source = nullptr;
        }

        return err;
    }

    err = clBuildProgram(ocl->program, 1, &ocl->device, "", nullptr, nullptr);
    if (CL_SUCCESS != err)
    {
        log_error("Error: clBuildProgram() for source program returned %s.\n", translate_open_cl_error(err));

        if (err == CL_BUILD_PROGRAM_FAILURE)
        {
            size_t log_size = 0;
            clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

            std::vector<char> build_log(log_size);
            clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, log_size, &build_log[0], nullptr);

            log_error("Error happened during the build of OpenCL program.\nBuild log:%s", &build_log[0]);
        }
    }

    if (source)
    {
        delete[] source;
        source = nullptr;
    }

    return err;
}

cl_uint set_kernel_arguments(ocl_args_d_t* ocl, cl_uint width, cl_uint height, cl_float temperature, char axis)
{
    auto err = CL_SUCCESS;

    err = clSetKernelArg(ocl->kernel, 0, sizeof(cl_mem), static_cast<void*>(&ocl->input));
    if (CL_SUCCESS != err)
    {
        log_error("error: Failed to set argument input, returned %s\n", translate_open_cl_error(err));
        return err;
    }

    err = clSetKernelArg(ocl->kernel, 1, sizeof(cl_mem), static_cast<void*>(&ocl->output));
    if (CL_SUCCESS != err)
    {
        log_error("Error: Failed to set argument output, returned %s\n", translate_open_cl_error(err));
        return err;
    }

    err = clSetKernelArg(ocl->kernel, 2, sizeof(cl_uint), static_cast<void*>(&width));
    if (CL_SUCCESS != err)
    {
        log_error("Error: Failed to set argument temperature, returned %s\n", translate_open_cl_error(err));
        return err;
    }

    err = clSetKernelArg(ocl->kernel, 3, sizeof(cl_uint), static_cast<void*>(&height));
    if (CL_SUCCESS != err)
    {
        log_error("Error: Failed to set argument temperature, returned %s\n", translate_open_cl_error(err));
        return err;
    }

    err = clSetKernelArg(ocl->kernel, 4, sizeof(cl_float), static_cast<void*>(&temperature));
    if (CL_SUCCESS != err)
    {
        log_error("Error: Failed to set argument temperature, returned %s\n", translate_open_cl_error(err));
        return err;
    }

    err = clSetKernelArg(ocl->kernel, 5, sizeof(char), static_cast<void*>(&axis));
    if (CL_SUCCESS != err)
    {
        log_error("Error: Failed to set argument temperature, returned %s\n", translate_open_cl_error(err));
        return err;
    }

    return err;
}

cl_uint execute_add_kernel(ocl_args_d_t* ocl, const cl_uint width, const cl_uint height)
{
    auto err = CL_SUCCESS;

    size_t global_work_size[2] = { width, height };

    err = clEnqueueNDRangeKernel(ocl->command_queue, ocl->kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
    if (CL_SUCCESS != err)
    {
        log_error("Error: Failed to run kernel, return %s\n", translate_open_cl_error(err));
        return err;
    }

    err = clFinish(ocl->command_queue);
    if (CL_SUCCESS != err)
    {
        log_error("Error: clFinish return %s\n", translate_open_cl_error(err));
        return err;
    }

    return CL_SUCCESS;
}

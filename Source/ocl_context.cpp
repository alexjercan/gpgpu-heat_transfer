#include "ocl_context.h"

#include <vector>


#include "log_utils.h"
#include "ocl_args.h"
#include "utils.h"

static cl_platform_id find_open_cl_platform(const char* preferred_platform, cl_device_type device_type);
static int get_platform_and_device_version(cl_platform_id platform_id, ocl_args_d_t* ocl);
bool check_preferred_platform_match(cl_platform_id platform, const char* preferred_platform);

int setup_open_cl(ocl_args_d_t* ocl, const cl_device_type device_type, const char* preferred_platform)
{
    auto err = CL_SUCCESS;

    auto* platform_id = find_open_cl_platform(preferred_platform, device_type);
    if (nullptr == platform_id)
    {
        log_error("Error: Failed to find OpenCL platform.\n");
        return CL_INVALID_VALUE;
    }

    cl_context_properties context_properties[] = { CL_CONTEXT_PLATFORM, cl_context_properties(platform_id), 0 };
    ocl->context = clCreateContextFromType(context_properties, device_type, nullptr, nullptr, &err);
    if ((CL_SUCCESS != err) || (nullptr == ocl->context))
    {
        log_error("Couldn't create a context, clCreateContextFromType() returned '%s'.\n", translate_open_cl_error(err));
        return err;
    }

    err = clGetContextInfo(ocl->context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &ocl->device, nullptr);
    if (CL_SUCCESS != err)
    {
        log_error("Error: clGetContextInfo() to get list of devices returned %s.\n", translate_open_cl_error(err));
        return err;
    }

    get_platform_and_device_version(platform_id, ocl);

#ifdef CL_VERSION_2_0
    if (OPENCL_VERSION_2_0 == ocl->device_version)
    {
        const cl_command_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
        ocl->command_queue = clCreateCommandQueueWithProperties(ocl->context, ocl->device, properties, &err);
    }
    else
    {
        const cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
        ocl->command_queue = clCreateCommandQueue(ocl->context, ocl->device, properties, &err);
    }
#else
    cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
    ocl->commandQueue = clCreateCommandQueue(ocl->context, ocl->device, properties, &err);
#endif

    if (CL_SUCCESS != err)
    {
        log_error("Error: clCreateCommandQueue() returned %s.\n", translate_open_cl_error(err));
        return err;
    }

    return CL_SUCCESS;
}

cl_platform_id find_open_cl_platform(const char* preferred_platform, const cl_device_type device_type)
{
    cl_uint num_platforms = 0;
    auto err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (CL_SUCCESS != err)
    {
        log_error("Error: clGetPlatformIDs() to get num platforms returned %s.\n", translate_open_cl_error(err));
        return nullptr;
    }

    if (0 == num_platforms)
    {
        log_error("Error: No platforms found!\n");
        return nullptr;
    }

    std::vector<cl_platform_id> platforms(num_platforms);

    err = clGetPlatformIDs(num_platforms, &platforms[0], nullptr);
    if (CL_SUCCESS != err)
    {
        log_error("Error: clGetPlatformIDs() to get platforms returned %s.\n", translate_open_cl_error(err));
        return nullptr;
    }

    for (cl_uint i = 0; i < num_platforms; i++)
    {
        auto match = true;
        cl_uint num_devices = 0;

        if (nullptr != preferred_platform && strlen(preferred_platform) > 0)
        {
            match = check_preferred_platform_match(platforms[i], preferred_platform);
        }

        if (match)
        {
            err = clGetDeviceIDs(platforms[i], device_type, 0, nullptr, &num_devices);
            if (CL_SUCCESS != err)
                log_error("clGetDeviceIDs() returned %s.\n", translate_open_cl_error(err));

            if (0 != num_devices)
                return platforms[i];
        }
    }

    return nullptr;
}

int get_platform_and_device_version(cl_platform_id platform_id, ocl_args_d_t* ocl)
{
    size_t string_length = 0;
	auto err = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, 0, nullptr, &string_length);

    if (CL_SUCCESS != err)
    {
        log_error("Error: clGetPlatformInfo() to get CL_PLATFORM_VERSION length returned '%s'.\n", translate_open_cl_error(err));
        return err;
    }

    std::vector<char> platform_version(string_length);

    err = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, string_length, &platform_version[0], nullptr);
    if (CL_SUCCESS != err)
    {
        log_error("Error: clGetPlatformInfo() to get CL_PLATFORM_VERSION returned %s.\n", translate_open_cl_error(err));
        return err;
    }

    if (strstr(&platform_version[0], "OpenCL 2.0") != nullptr)
    {
        ocl->platform_version = OPENCL_VERSION_2_0;
    }

    err = clGetDeviceInfo(ocl->device, CL_DEVICE_VERSION, 0, nullptr, &string_length);
    if (CL_SUCCESS != err)
    {
        log_error("Error: clGetDeviceInfo() to get CL_DEVICE_VERSION length returned '%s'.\n", translate_open_cl_error(err));
        return err;
    }

    std::vector<char> device_version(string_length);

    err = clGetDeviceInfo(ocl->device, CL_DEVICE_VERSION, string_length, &device_version[0], nullptr);
    if (CL_SUCCESS != err)
    {
        log_error("Error: clGetDeviceInfo() to get CL_DEVICE_VERSION returned %s.\n", translate_open_cl_error(err));
        return err;
    }

    if (strstr(&device_version[0], "OpenCL 2.0") != nullptr)
    {
        ocl->device_version = OPENCL_VERSION_2_0;
    }

    err = clGetDeviceInfo(ocl->device, CL_DEVICE_OPENCL_C_VERSION, 0, nullptr, &string_length);
    if (CL_SUCCESS != err)
    {
        log_error("Error: clGetDeviceInfo() to get CL_DEVICE_OPENCL_C_VERSION length returned '%s'.\n", translate_open_cl_error(err));
        return err;
    }

    std::vector<char> compiler_version(string_length);

    err = clGetDeviceInfo(ocl->device, CL_DEVICE_OPENCL_C_VERSION, string_length, &compiler_version[0], nullptr);
    if (CL_SUCCESS != err)
    {
        log_error("Error: clGetDeviceInfo() to get CL_DEVICE_OPENCL_C_VERSION returned %s.\n", translate_open_cl_error(err));
        return err;
    }

    if (strstr(&compiler_version[0], "OpenCL C 2.0") != nullptr)
    {
        ocl->compiler_version = OPENCL_VERSION_2_0;
    }

    return err;
}

bool check_preferred_platform_match(cl_platform_id platform, const char* preferred_platform)
{
    size_t string_length = 0;
    auto err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &string_length);
    auto match = false;

    if (CL_SUCCESS != err)
    {
        log_error("Error: clGetPlatformInfo() to get CL_PLATFORM_NAME length returned '%s'.\n", translate_open_cl_error(err));
        return false;
    }

    std::vector<char> platform_name(string_length);

    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, string_length, &platform_name[0], nullptr);
    if (CL_SUCCESS != err)
    {
        log_error("Error: clGetPlatformInfo() to get CL_PLATFORM_NAME returned %s.\n", translate_open_cl_error(err));
        return false;
    }

    if (strstr(&platform_name[0], preferred_platform) != nullptr)
        match = true;

    return match;
}

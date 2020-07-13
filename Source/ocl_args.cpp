#include "ocl_args.h"

#include <stdlib.h>
#include <tchar.h>
#include <vector>

#include "CL/cl.h"

#include <Windows.h>


#include "log_utils.h"
#include "../HeatTransfer/HeatTransfer/utils.h"

#define OPENCL_VERSION_1_2  1.2f
#define OPENCL_VERSION_2_0  2.0f

const char* translate_open_cl_error(const cl_int error_code)
{
    switch (error_code)
    {
    case CL_SUCCESS:                            return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:                   return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:               return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:             return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:                   return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:                 return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:       return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:                   return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH:              return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE:              return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:                        return "CL_MAP_FAILURE";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:       return "CL_MISALIGNED_SUB_BUFFER_OFFSET";                          
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";   
    case CL_COMPILE_PROGRAM_FAILURE:            return "CL_COMPILE_PROGRAM_FAILURE";                               
    case CL_LINKER_NOT_AVAILABLE:               return "CL_LINKER_NOT_AVAILABLE";                                  
    case CL_LINK_PROGRAM_FAILURE:               return "CL_LINK_PROGRAM_FAILURE";                                  
    case CL_DEVICE_PARTITION_FAILED:            return "CL_DEVICE_PARTITION_FAILED";                               
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";                         
    case CL_INVALID_VALUE:                      return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:                return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:                   return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:                     return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:                    return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:           return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:              return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:                   return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:                 return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE:                 return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER:                    return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY:                     return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:              return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:                    return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:         return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:                return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:          return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:                     return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:                  return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:                  return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:                   return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:                return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:             return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:            return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:             return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:              return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:            return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:                      return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:                  return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT:                  return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE:                return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL:                  return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE:           return "CL_INVALID_GLOBAL_WORK_SIZE";                           
    case CL_INVALID_PROPERTY:                   return "CL_INVALID_PROPERTY";                                   
    case CL_INVALID_IMAGE_DESCRIPTOR:           return "CL_INVALID_IMAGE_DESCRIPTOR";                           
    case CL_INVALID_COMPILER_OPTIONS:           return "CL_INVALID_COMPILER_OPTIONS";                           
    case CL_INVALID_LINKER_OPTIONS:             return "CL_INVALID_LINKER_OPTIONS";                             
    case CL_INVALID_DEVICE_PARTITION_COUNT:     return "CL_INVALID_DEVICE_PARTITION_COUNT";                     
    case CL_INVALID_PIPE_SIZE:                  return "CL_INVALID_PIPE_SIZE";                                  
    case CL_INVALID_DEVICE_QUEUE:               return "CL_INVALID_DEVICE_QUEUE";                                   

    default:
        return "UNKNOWN ERROR CODE";
    }
}

ocl_args_d_t::ocl_args_d_t() :
    context(nullptr),
    device(nullptr),
    command_queue(nullptr),
    program(nullptr),
    kernel(nullptr),
    platform_version(OPENCL_VERSION_1_2),
    device_version(OPENCL_VERSION_1_2),
    compiler_version(OPENCL_VERSION_1_2),
    srcA(nullptr),
    srcB(nullptr),
    dstMem(nullptr)
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
    if (srcA)
    {
        err = clReleaseMemObject(srcA);
        if (CL_SUCCESS != err)
	        log_error("Error: clReleaseMemObject returned '%s'.\n", translate_open_cl_error(err));
    }
    if (srcB)
    {
        err = clReleaseMemObject(srcB);
        if (CL_SUCCESS != err)
	        log_error("Error: clReleaseMemObject returned '%s'.\n", translate_open_cl_error(err));
    }
    if (dstMem)
    {
        err = clReleaseMemObject(dstMem);
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


bool check_preferred_platform_match(cl_platform_id platform, const char* preferred_platform)
{
    size_t string_length = 0;
    auto err = CL_SUCCESS;
    auto match = false;

    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &string_length);
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

cl_platform_id find_open_cl_platform(const char* preferred_platform, const cl_device_type device_type)
{
    cl_uint num_platforms = 0;
    auto err = CL_SUCCESS;

    err = clGetPlatformIDs(0, nullptr, &num_platforms);
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
	auto err = CL_SUCCESS;

    size_t string_length = 0;
    err = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, 0, nullptr, &string_length);
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


/*
 * Generate random value for input buffers
 */
void generateInput(cl_int* inputArray, cl_uint arrayWidth, cl_uint arrayHeight)
{
    srand(12345);

    // random initialization of input
    cl_uint array_size = arrayWidth * arrayHeight;
    for (cl_uint i = 0; i < array_size; ++i)
    {
        inputArray[i] = rand();
    }
}

int setup_open_cl(ocl_args_d_t* ocl, const cl_device_type device_type)
{
	auto err = CL_SUCCESS;

	cl_platform_id platform_id = find_open_cl_platform("Intel", device_type);
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

int create_and_build_program(ocl_args_d_t* ocl)
{
	auto err = CL_SUCCESS;

    char* source = nullptr;
    size_t src_size = 0;
    err = ReadSourceFromFile("Template.cl", &source, &src_size);
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


int create_buffer_arguments(ocl_args_d_t* ocl, cl_int* input_a, cl_int* input_b, cl_int* output_c, const cl_uint array_width, const cl_uint array_height)
{
	auto err = CL_SUCCESS;

    cl_image_format format;
    cl_image_desc desc;

    format.image_channel_data_type = CL_UNSIGNED_INT32;
    format.image_channel_order = CL_R;

    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = array_width;
    desc.image_height = array_height;
    desc.image_depth = 0;
    desc.image_array_size = 1;
    desc.image_row_pitch = 0;
    desc.image_slice_pitch = 0;
    desc.num_mip_levels = 0;
    desc.num_samples = 0;
	
#ifdef CL_VERSION_2_0
    desc.mem_object = nullptr;
#else
    desc.buffer = nullptr;
#endif

    ocl->srcA = clCreateImage(ocl->context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, &format, &desc, input_a, &err);
    if (CL_SUCCESS != err)
    {
        log_error("Error: clCreateImage for srcA returned %s\n", translate_open_cl_error(err));
        return err;
    }

    ocl->srcB = clCreateImage(ocl->context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, &format, &desc, input_b, &err);
    if (CL_SUCCESS != err)
    {
        log_error("Error: clCreateImage for srcB returned %s\n", translate_open_cl_error(err));
        return err;
    }

    ocl->dstMem = clCreateImage(ocl->context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, &format, &desc, output_c, &err);
    if (CL_SUCCESS != err)
    {
        log_error("Error: clCreateImage for dstMem returned %s\n", translate_open_cl_error(err));
        return err;
    }


    return CL_SUCCESS;
}

cl_uint set_kernel_arguments(ocl_args_d_t* ocl)
{
	auto err = CL_SUCCESS;

    err = clSetKernelArg(ocl->kernel, 0, sizeof(cl_mem), static_cast<void*>(&ocl->srcA));
    if (CL_SUCCESS != err)
    {
        log_error("error: Failed to set argument srcA, returned %s\n", translate_open_cl_error(err));
        return err;
    }

    err = clSetKernelArg(ocl->kernel, 1, sizeof(cl_mem), static_cast<void*>(&ocl->srcB));
    if (CL_SUCCESS != err)
    {
        log_error("Error: Failed to set argument srcB, returned %s\n", translate_open_cl_error(err));
        return err;
    }

    err = clSetKernelArg(ocl->kernel, 2, sizeof(cl_mem), static_cast<void*>(&ocl->dstMem));
    if (CL_SUCCESS != err)
    {
        log_error("Error: Failed to set argument dstMem, returned %s\n", translate_open_cl_error(err));
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

bool read_and_verify(ocl_args_d_t* ocl, const cl_uint width, const cl_uint height, cl_int* input_a, cl_int* input_b)
{
	auto err = CL_SUCCESS;
	auto result = true;

    size_t origin[] = { 0, 0, 0 };
    size_t region[] = { width, height, 1 };
    size_t image_row_pitch;
    size_t image_slice_pitch;
	auto* const result_ptr = static_cast<cl_int*>(clEnqueueMapImage(ocl->command_queue, ocl->dstMem, true, CL_MAP_READ, origin, region,
	                                                                &image_row_pitch, &image_slice_pitch, 0, nullptr, nullptr, &err));

    if (CL_SUCCESS != err)
    {
        log_error("Error: clEnqueueMapBuffer returned %s\n", translate_open_cl_error(err));
        return false;
    }

    err = clFinish(ocl->command_queue);
    if (CL_SUCCESS != err)
    {
        log_error("Error: clFinish returned %s\n", translate_open_cl_error(err));
    }

	const auto size = width * height;
    for (unsigned int k = 0; k < size; ++k)
    {
        if (result_ptr[k] != input_a[k] + input_b[k])
        {
            log_error("Verification failed at %d: (%d + %d = %d)\n", k, input_a[k], input_b[k], result_ptr[k]);
            result = false;
        }
    }

    err = clEnqueueUnmapMemObject(ocl->command_queue, ocl->dstMem, result_ptr, 0, nullptr, nullptr);
    if (CL_SUCCESS != err)
    {
        log_error("Error: clEnqueueUnmapMemObject() returned %s\n", translate_open_cl_error(err));
    }

    return result;
}

int _tmain(int argc, TCHAR* argv[])
{
    cl_int err;
    ocl_args_d_t ocl;
    const cl_device_type device_type = CL_DEVICE_TYPE_GPU;

    LARGE_INTEGER perfFrequency;
    LARGE_INTEGER performanceCountNDRangeStart;
    LARGE_INTEGER performanceCountNDRangeStop;

    const cl_uint array_width = 1024;
    const cl_uint array_height = 1024;

    if (CL_SUCCESS != setup_open_cl(&ocl, device_type))
    {
        return -1;
    }

    // allocate working buffers. 
    // the buffer should be aligned with 4K page and size should fit 64-byte cached line
    cl_uint optimizedSize = ((sizeof(cl_int) * array_width * array_height - 1) / 64 + 1) * 64;
    cl_int* inputA = (cl_int*)_aligned_malloc(optimizedSize, 4096);
    cl_int* inputB = (cl_int*)_aligned_malloc(optimizedSize, 4096);
    cl_int* outputC = (cl_int*)_aligned_malloc(optimizedSize, 4096);
    if (NULL == inputA || NULL == inputB || NULL == outputC)
    {
        log_error("Error: _aligned_malloc failed to allocate buffers.\n");
        return -1;
    }

    //random input
    generateInput(inputA, array_width, array_height);
    generateInput(inputB, array_width, array_height);

    if (CL_SUCCESS != create_buffer_arguments(&ocl, inputA, inputB, outputC, array_width, array_height))
    {
        return -1;
    }

    if (CL_SUCCESS != create_and_build_program(&ocl))
    {
        return -1;
    }

    ocl.kernel = clCreateKernel(ocl.program, "Add", &err);
    if (CL_SUCCESS != err)
    {
        log_error("Error: clCreateKernel returned %s\n", translate_open_cl_error(err));
        return -1;
    }

    if (CL_SUCCESS != set_kernel_arguments(&ocl))
    {
        return -1;
    }

    // Regularly you wish to use OpenCL in your application to achieve greater performance results
    // that are hard to achieve in other ways.
    // To understand those performance benefits you may want to measure time your application spent in OpenCL kernel execution.
    // The recommended way to obtain this time is to measure interval between two moments:
    //   - just before clEnqueueNDRangeKernel is called, and
    //   - just after clFinish is called
    // clFinish is necessary to measure entire time spending in the kernel, measuring just clEnqueueNDRangeKernel is not enough,
    // because this call doesn't guarantees that kernel is finished.
    // clEnqueueNDRangeKernel is just enqueue new command in OpenCL command queue and doesn't wait until it ends.
    // clFinish waits until all commands in command queue are finished, that suits your need to measure time.
    bool queueProfilingEnable = true;
    if (queueProfilingEnable)
        QueryPerformanceCounter(&performanceCountNDRangeStart);
    // Execute (enqueue) the kernel
    if (CL_SUCCESS != execute_add_kernel(&ocl, array_width, array_height))
    {
        return -1;
    }
    if (queueProfilingEnable)
        QueryPerformanceCounter(&performanceCountNDRangeStop);

    // The last part of this function: getting processed results back.
    // use map-unmap sequence to update original memory area with output buffer.
    read_and_verify(&ocl, array_width, array_height, inputA, inputB);

    // retrieve performance counter frequency
    if (queueProfilingEnable)
    {
        QueryPerformanceFrequency(&perfFrequency);
        log_info("NDRange performance counter time %f ms.\n",
            1000.0f * (float)(performanceCountNDRangeStop.QuadPart - performanceCountNDRangeStart.QuadPart) / (float)perfFrequency.QuadPart);
    }

    _aligned_free(inputA);
    _aligned_free(inputB);
    _aligned_free(outputC);

    return 0;
}


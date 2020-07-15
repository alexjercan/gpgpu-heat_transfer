#include "cl_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "../../log_utils.h"

cl_int parse_arguments(const int argc, char** argv, cl_uint* num_dimensions, size_t* global_work_items,
                       size_t* local_work_items, int* next_arg_index)
{
	// Order of arguments:
	// argv[1] - number of dimensions for the global and local work item lists
	// argv[2] .. argv[argv[1] + 1] - number of global work items on each dimension (optional)
	// argv[argv[1] + 2] .. argv[2 * argv[1] + 1] - number of local work items on each dimension
	int i;

	CHECK_VALUE(argc <= 1, CL_INVALID_ARG_VALUE);
	CHECK_VALUE(!local_work_items, CL_INVALID_ARG_VALUE);
	const cl_uint dimensions = atoi(argv[1]);

	// The number of allowed dimensions is 1..3
	SAFE_PTR_SET(num_dimensions, dimensions);
	CHECK_VALUE((dimensions < 1) || (dimensions > 3), CL_INVALID_VALUE);

	if (global_work_items)
	{
		CHECK_VALUE(argc < 2 + 2 * dimensions, CL_INVALID_VALUE);
	}
	else
	{
		CHECK_VALUE(argc < 2 + dimensions, CL_INVALID_VALUE);
	}

	// Read the global and local work sizes for each dimension
	for (i = 0; i < dimensions; ++i)
	{
		if (global_work_items)
		{
			global_work_items[i] = atoi(argv[i + 2]);
			local_work_items[i] = atoi(argv[dimensions + i + 2]);
		}
		else
		{
			local_work_items[i] = atoi(argv[i + 2]);
		}
	}

	log_info("OpenCL execution configuration: " NEW_LINE);
	log_info("- Dimensions: %d" NEW_LINE, dimensions);

	if (global_work_items)
	{
		log_info("- Global work items: (");

		// Print the global and local work sizes
		for (i = 0; i < dimensions - 1; ++i)
		{
			log_info("%zu, ", global_work_items[i]);
		}
		log_info("%zu)%s", global_work_items[dimensions - 1], NEW_LINE);
	}

	log_info("- Local work items: (");

	for (i = 0; i < dimensions - 1; ++i)
	{
		log_info("%zu, ", local_work_items[i]);
	}
	log_info("%zu)" NEW_LINE, local_work_items[dimensions - 1]);

	// Output an index to the remaining (not parsed) command arguments
	SAFE_PTR_SET(next_arg_index, global_work_items ? 2 * dimensions + 2 : dimensions + 2);

	return CL_SUCCESS;
}

cl_int get_device_version(cl_device_id device, int* major, int* minor)
{
	char device_info[256], * str = nullptr;
	int dev_major, dev_minor, drv_major, drv_minor;

	SAFE_PTR_SET(major, -1);
	SAFE_PTR_SET(minor, -1);
	CHECK_VALUE(!device, CL_INVALID_VALUE);

	// The OpenCL device provides a version for both itself and the driver
	SAFE_OCL_CALL(clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(device_info) - 1, device_info, NULL));
	str = strchr(device_info, ' ');
	CHECK_VALUE(!str, CL_INVALID_VALUE);
	CHECK_VALUE(sscanf(str + 1, "%d.%d", &dev_major, &dev_minor) < 2, CL_INVALID_VALUE);

	SAFE_OCL_CALL(clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(device_info) - 1, device_info, NULL));
	CHECK_VALUE(sscanf(device_info, "%d.%d", &drv_major, &drv_minor) < 2, CL_INVALID_VALUE);

	// Get the smallest (oldest) supported OpenCL version
	SAFE_PTR_SET(major, drv_major < dev_major ? drv_major : dev_major);
	SAFE_PTR_SET(minor, drv_minor < dev_minor ? drv_minor : dev_minor);

	return CL_SUCCESS;
}

cl_int init_context(cl_device_type dev_type, cl_platform_id* platform, cl_device_id* device, cl_context* context)
{
	cl_context_properties context_properties[] = { CL_CONTEXT_PLATFORM, 0, 0 };
	cl_platform_id* platforms = nullptr;
	cl_device_id* devices = nullptr;
	cl_uint num_platforms, num_devices;
	auto status = CL_SUCCESS;

	// First, get a list of available OpenCL platforms on the current system
	CHECK_VALUE(!context, CL_INVALID_VALUE);
	SAFE_OCL_CALL(clGetPlatformIDs(0, NULL, &num_platforms));
	CHECK_VALUE(!num_platforms, CL_INVALID_VALUE);

	platforms = static_cast<cl_platform_id*>(malloc(num_platforms * sizeof(cl_platform_id)));
	SAFE_OCL_CALL(clGetPlatformIDs(num_platforms, platforms, NULL));

	for (auto i = 0u; i < num_platforms; ++i)
	{
		// Next, get a list of devices with the specified type for each available platform
		SAFE_OCL_CALL(clGetDeviceIDs(platforms[i], dev_type, 0, NULL, &num_devices));

		if (num_devices == 0)
			continue;

		SAFE_FREE(devices);
		devices = static_cast<cl_device_id*>(malloc(num_devices * sizeof(cl_device_id)));
		SAFE_OCL_CALL(clGetDeviceIDs(platforms[i], dev_type, num_devices, devices, NULL));

		// Attempt to create a context with the current platform and the first found device
		context_properties[1] = reinterpret_cast<cl_context_properties>(platforms[i]);
		SAFE_PTR_SET(context, clCreateContext(context_properties, 1, devices, nullptr, NULL, &status));

		// Stop when a context was created successfully
		if (status == CL_SUCCESS)
		{
			SAFE_PTR_SET(platform, platforms[i]);
			SAFE_PTR_SET(device, devices[0]);
			break;
		}
	}

	// Free used resources
	SAFE_FREE(devices);
	SAFE_FREE(platforms);

	return CL_SUCCESS;
}

cl_int get_program_from_file(const char* file_path, cl_context context, cl_program* program)
{
	size_t source_size;
	char* file_source;
	cl_device_id device;
	cl_int status;

	CHECK_VALUE(!context || !program, CL_INVALID_VALUE);

	// Load the OpenCL program source code from the file
	auto* file_ptr = fopen(file_path, "r");
	if (!file_ptr)
	{
		log_info("Error opening OpenCL source file <%s>!" NEW_LINE, file_path);
		return CL_INVALID_VALUE;
	}

	// Read the full contents of the file into a host memory buffer
	fseek(file_ptr, 0L, SEEK_END);
	const auto file_size = ftell(file_ptr) + 1;
	rewind(file_ptr);

	file_source = static_cast<char*>(calloc(file_size, 1));
	source_size = fread(file_source, 1, file_size, file_ptr);
	fclose(file_ptr);

	// Create an OpenCL program with the read source code and compile it (using on-line compilation)
	SAFE_PTR_SET(program, clCreateProgramWithSource(context, 1, const_cast<const char**>(&file_source), static_cast<const size_t*>(&source_size), &status));
	SAFE_FREE(file_source);

	CHECK_VALUE(status != CL_SUCCESS, status);
	SAFE_OCL_CALL(clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &device, NULL));

	// Actual program compilation
	status = clBuildProgram(*program, 1, &device, nullptr, nullptr, nullptr);

	if (status != CL_SUCCESS)
	{
		size_t log_length;

		// If the compilation was not successful, display the compiler messages
		log_info("Error building the OpenCL program from <%s>!" NEW_LINE, file_path);

		SAFE_OCL_CALL(clGetProgramBuildInfo(*program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_length));
		++log_length;

		auto* log_buffer = static_cast<char*>(calloc(log_length, 1));
		SAFE_OCL_CALL(clGetProgramBuildInfo(*program, device, CL_PROGRAM_BUILD_LOG, log_length, log_buffer, NULL));

		log_info("Build log:%s<%s>" NEW_LINE, NEW_LINE, log_buffer);
		SAFE_FREE(log_buffer);

		return status;
	}

	return CL_SUCCESS;
}

cl_int get_kernel(cl_program program, const char* kernel_name, cl_kernel* kernel)
{
	cl_int status;

	CHECK_VALUE(!program || !kernel_name || !kernel, CL_INVALID_VALUE);
	SAFE_PTR_SET(kernel, clCreateKernel(program, kernel_name, &status));

	return status;
}

cl_int create_command_queue(cl_context context, cl_command_queue* queue)
{
	cl_device_id device = nullptr;
	cl_int status;

	CHECK_VALUE(!context || !queue, CL_INVALID_VALUE);

	// Read the context's device and use it when creating the queue
	SAFE_OCL_CALL(clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &device, NULL));
	SAFE_PTR_SET(queue, clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status));

	return status;
}

cl_int create_memory_buffer(cl_context context, const cl_mem_flags flags, const size_t size, void* host_ptr, cl_mem* buffer)
{
	auto status = CL_SUCCESS;

	CHECK_VALUE(!context, CL_INVALID_VALUE);
	SAFE_PTR_SET(buffer, clCreateBuffer(context, flags, size, host_ptr, &status));
	return status;
}

cl_int generate_host_array(const cl_uint num_elements, const cl_float initial_value, float** array)
{
	CHECK_VALUE(!array || !num_elements, CL_INVALID_VALUE);

	SAFE_PTR_SET(array, static_cast<float*>(malloc(num_elements * sizeof(float))));
	CHECK_VALUE(!(*array), CL_INVALID_VALUE);

	for (cl_uint i = 0; i < num_elements; ++i)
		(*array)[i] = initial_value;

	return CL_SUCCESS;
}

size_t get_worker_count(const cl_uint num_dimensions, const size_t* global_work_items)
{
	size_t workers = 1;

	CHECK_VALUE(!global_work_items, 0);

	for (size_t i = 0; i < num_dimensions; ++i)
	{
		workers *= global_work_items[i];
	}

	return workers;
}

const char* get_error_string(const cl_int status)
{
	switch (status)
	{
	case CL_SUCCESS:                            return "Success";
	case CL_DEVICE_NOT_FOUND:                   return "Device not found";
	case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
	case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
	case CL_OUT_OF_RESOURCES:                   return "Out of resources";
	case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
	case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
	case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
	case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
	case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
	case CL_MAP_FAILURE:                        return "Map failure";
	case CL_INVALID_VALUE:                      return "Invalid value";
	case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
	case CL_INVALID_PLATFORM:                   return "Invalid platform";
	case CL_INVALID_DEVICE:                     return "Invalid device";
	case CL_INVALID_CONTEXT:                    return "Invalid context";
	case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
	case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
	case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
	case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
	case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
	case CL_INVALID_SAMPLER:                    return "Invalid sampler";
	case CL_INVALID_BINARY:                     return "Invalid binary";
	case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
	case CL_INVALID_PROGRAM:                    return "Invalid program";
	case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
	case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
	case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
	case CL_INVALID_KERNEL:                     return "Invalid kernel";
	case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
	case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
	case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
	case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
	case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
	case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
	case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
	case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
	case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
	case CL_INVALID_EVENT:                      return "Invalid event";
	case CL_INVALID_OPERATION:                  return "Invalid operation";
	case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
	case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
	case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
	default:                                    return "Unknown error";
	}
}

void calibrate_global_work_size(const int major, int minor, const cl_uint num_dimensions, size_t* global_work_items,
                                const size_t* local_work_items, size_t* calibrated_global_work_items)
{
	// No calibration needed for OpenCL 2.0 and above
	if (major >= 2)
	{
		if (global_work_items && calibrated_global_work_items)
		{
			memcpy(calibrated_global_work_items, global_work_items, num_dimensions * sizeof(size_t));
		}

		return;
	}

	// For older versions, the global work size must be a multiple of the local work size on each dimension
	for (size_t i = 0; i < num_dimensions; ++i)
	{
		calibrated_global_work_items[i] = ((global_work_items[i] + local_work_items[i] - 1) / local_work_items[i]) * local_work_items[i];
	}
}

cl_int print_platform_info(cl_platform_id platform)
{
	// The platform info items to query
	const cl_platform_info cl_platform_attributes[] = {
		CL_PLATFORM_NAME, 		CL_PLATFORM_VENDOR,
		CL_PLATFORM_VERSION, 	CL_PLATFORM_PROFILE,
		CL_PLATFORM_EXTENSIONS };
	const char* cl_platform_attributes_names[] = {
		"Name", "Vendor", "Version", "Profile", "Extensions" };
	size_t info_size = 0;

	CHECK_VALUE(!platform, CL_INVALID_VALUE);
	log_info("OpenCL platform <%p>:" NEW_LINE, platform);

	// Get platform info items and display them
	for (size_t i = 0; i < sizeof(cl_platform_attributes) / sizeof(cl_platform_attributes[0]); ++i)
	{
		SAFE_OCL_CALL(clGetPlatformInfo(platform, cl_platform_attributes[i], 0, NULL, &info_size));
		auto* info = static_cast<char*>(malloc(info_size + 1));

		SAFE_OCL_CALL(clGetPlatformInfo(platform, cl_platform_attributes[i], info_size, info, NULL));
		log_info("- %s: %s" NEW_LINE, cl_platform_attributes_names[i], info);
	}

	return CL_SUCCESS;
}

cl_int print_device_info(cl_device_id device)
{
	// The device info items to query
	const cl_device_info cl_device_infos[] = {
		CL_DEVICE_NAME,		CL_DEVICE_VENDOR,
		CL_DEVICE_VERSION,	CL_DRIVER_VERSION,
		CL_DEVICE_PROFILE,	CL_DEVICE_EXTENSIONS };
	const char* cl_device_infos_names[] = {
		"Name", "Vendor", "Version", "Driver version", "Profile", "Extensions" };
	size_t info_size = 0;
	cl_device_type type_data = 0;
	cl_ulong ulong_data = 0;
	cl_uint uint_data = 0;
	cl_bool bool_data = 0;
	size_t size_data = 0;

	CHECK_VALUE(!device, CL_INVALID_VALUE);
	log_info("OpenCL device <%p>: " NEW_LINE, device);

	// Get the device type (CPU, GPU, accelerator or default)
	SAFE_OCL_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &type_data, NULL));
	switch (type_data)
	{
		case CL_DEVICE_TYPE_CPU: 			PRINT_DEVICE_TYPE("CPU");
		case CL_DEVICE_TYPE_GPU: 			PRINT_DEVICE_TYPE("GPU");
		case CL_DEVICE_TYPE_ACCELERATOR: 	PRINT_DEVICE_TYPE("Accelerator");
		default: 							PRINT_DEVICE_TYPE("Default");
	}

	// Get the device info strings and display them
	for (auto i = 0; i < sizeof(cl_device_infos) / sizeof(cl_device_infos[0]); ++i)
	{
		SAFE_OCL_CALL(clGetDeviceInfo(device, cl_device_infos[i], 0, NULL, &info_size));
		auto* info = static_cast<char*>(malloc(info_size + 1));

		SAFE_OCL_CALL(clGetDeviceInfo(device, cl_device_infos[i], info_size, info, NULL));
		log_info("- %s: %s" NEW_LINE, cl_device_infos_names[i], info);
		SAFE_FREE(info);
	}

	// Get additional device information (global memory size, compute unites, compiler availability)
	SAFE_OCL_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &ulong_data, NULL));
	log_info("- Global memory size: %lu MB" NEW_LINE, ulong_data >> 20);

	SAFE_OCL_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &uint_data, NULL));
	log_info("- Available compute units: %d" NEW_LINE, uint_data);

	SAFE_OCL_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &size_data, NULL));
	log_info("- Maximum device work group size: %lu" NEW_LINE, size_data);

	SAFE_OCL_CALL(clGetDeviceInfo(device, CL_DEVICE_COMPILER_AVAILABLE, sizeof(cl_bool), &bool_data, NULL));
	log_info("- Compiler available: %s" NEW_LINE, bool_data ? "Yes" : "No");

	return CL_SUCCESS;
}

cl_int create_image(cl_context context, const cl_mem_flags mem_flags, const cl_uint width, const cl_uint height, cl_mem* image)
{
	cl_image_format image_format;
	cl_image_desc image_desc;
	auto status = CL_SUCCESS;

	CHECK_VALUE(!context || !image, CL_INVALID_VALUE);

	image_format.image_channel_order = CL_RGBA;
	image_format.image_channel_data_type = CL_UNORM_INT8;

	image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	image_desc.image_width = width;
	image_desc.image_height = height;
	image_desc.image_row_pitch = 0;
	image_desc.image_slice_pitch = 0;
	image_desc.num_mip_levels = 0;
	image_desc.num_samples = 0;
	image_desc.buffer = nullptr;

	SAFE_PTR_SET(image, clCreateImage(context, mem_flags, &image_format, &image_desc, NULL, &status));

	return status;
}

cl_int create_sampler(cl_context context, cl_sampler* sampler)
{
	cl_int status;

	CHECK_VALUE(!context || !sampler, CL_INVALID_VALUE);
	SAFE_PTR_SET(sampler, clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &status));

	return status;
}
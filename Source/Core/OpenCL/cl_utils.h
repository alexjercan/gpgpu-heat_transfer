#pragma once

#include <CL/cl.h>

#define NEW_LINE 					"\n"

#define MAX_WORK_DIMENSIONS			3

#define SAFE_PTR_SET(ptr, value)	do { if (ptr) (* (ptr)) = (value); } while (0)

#define SAFE_FREE(ptr)				do { if (ptr) free(ptr); } while (0)

#define CHECK_VALUE(value, status)	do { if (value) { log_info("Condition match: \"%s\" (%s:%d)"	\
														NEW_LINE, #value, __FILE__, __LINE__);	\
													  return (status); } } while (0)

#define SAFE_OCL_CALL(call)			do { cl_int status = (call); if (status != CL_SUCCESS) { 	\
										log_error("Error calling \"%s\" (%s:%d): %s" NEW_LINE,		\
											#call, __FILE__, __LINE__, get_error_string(status));	\
										return status; } } while (0)

#define PRINT_DEVICE_TYPE(type_str) { log_info("- Device type: %s" NEW_LINE, (type_str)); break; }

#define OPENCL_VERSION_1_2  1.2f
#define OPENCL_VERSION_2_0  2.0f

cl_int parse_arguments(int argc, char** argv, cl_uint * num_dimensions, size_t * global_work_items, size_t * local_work_items, int* next_arg_index);
cl_int get_device_version(cl_device_id device, int* major, int* minor);
cl_int init_context(cl_device_type dev_type, cl_platform_id* platform, cl_device_id* device, cl_context* context);
cl_int get_program_from_file(const char* file_path, cl_context context, cl_program* program);
cl_int get_kernel(cl_program program, const char* kernel_name, cl_kernel* kernel);
cl_int create_command_queue(cl_context context, cl_command_queue* queue);
cl_int create_memory_buffer(cl_context context, cl_mem_flags flags, size_t size, void* host_ptr, cl_mem* buffer);
cl_int generate_host_array(cl_uint num_elements, cl_float initial_value, float** array);

size_t get_worker_count(cl_uint num_dimensions, const size_t* global_work_items);
const char* get_error_string(cl_int status);
void calibrate_global_work_size(int major, int minor, cl_uint num_dimensions, size_t* global_work_items, const size_t* local_work_items, size_t* calibrated_global_work_items);

cl_int print_platform_info(cl_platform_id platform);
cl_int print_device_info(cl_device_id device);

cl_int create_image(cl_context context, cl_mem_flags mem_flags, cl_uint width, cl_uint height, cl_mem * image);
cl_int create_sampler(cl_context context, cl_sampler * sampler);
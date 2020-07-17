#pragma once
#include "CL/cl.h"
#include <d3d9.h>

#define OPENCL_VERSION_1_2  1.2f
#define OPENCL_VERSION_2_0  2.0f
#define INTEL_PLATFORM "Intel"
#define AMD_PLATFORM "AMD"

#define NEW_LINE 					"\n"

#define SAFE_OCL_CALL(call)			do { cl_int status = (call); if (status != CL_SUCCESS) { 	\
										log_error("Error calling \"%s\" (%s:%d): %s" NEW_LINE,		\
											#call, __FILE__, __LINE__, translate_open_cl_error(status));	\
										return status; } } while (0)

int read_source_from_file(const char* file_name, char** source, size_t* source_size);
void log_device_info(cl_device_id device);
void read_config(const char* input_file, const char*& preferred_platform, cl_uint & array_width, cl_uint & array_height, cl_float & plate_initial_temperature, float& air_temperature, float& point_temperature);




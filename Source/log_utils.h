#pragma once
#include <CL/cl.h>

const char* translate_open_cl_error(const cl_int error_code);

void log_info(const char* str, ...);

void log_error(const char* str, ...);
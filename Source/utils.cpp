#include <stdio.h>
#include <tchar.h>
#include "CL/cl.h"
#include "utils.h"

#include "log_utils.h"

//we want to use POSIX functions
#pragma warning( push )
#pragma warning( disable : 4996 )

int read_source_from_file(const char* file_name, char** source, size_t* source_size)
{
	auto error_code = CL_SUCCESS;

    FILE* fp = nullptr;
    fopen_s(&fp, file_name, "rb");
    if (fp == nullptr)
    {
        log_error("Error: Couldn't find program source file '%s'.\n", file_name);
        error_code = CL_INVALID_VALUE;
    }
    else {
        fseek(fp, 0, SEEK_END);
        *source_size = ftell(fp);
        fseek(fp, 0, SEEK_SET);

        *source = new char[*source_size];
        if (*source == nullptr)
        {
            log_error("Error: Couldn't allocate %d bytes for program source from file '%s'.\n", *source_size, file_name);
            error_code = CL_OUT_OF_HOST_MEMORY;
        }
        else {
            fread(*source, 1, *source_size, fp);
        }
    }
    return error_code;
}
#pragma warning( pop )
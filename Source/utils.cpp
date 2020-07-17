#include <stdio.h>
#include <tchar.h>
#include "CL/cl.h"
#include "utils.h"


#include <fstream>
#include <iosfwd>
#include <sstream>
#include <string>



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

void log_device_info(cl_device_id device)
{
    size_t info_size = 0;
    const cl_device_info cl_device_infos[] = {
        CL_DEVICE_NAME,		CL_DEVICE_VENDOR,
        CL_DEVICE_VERSION,	CL_DRIVER_VERSION,
        CL_DEVICE_PROFILE,	CL_DEVICE_EXTENSIONS };
    const char* cl_device_infos_names[] = { "Name", "Vendor", "Version", "Driver version", "Profile", "Extensions" };
    for (auto i = 0; i < 4; ++i)
    {
        clGetDeviceInfo(device, cl_device_infos[i], 0, nullptr, &info_size);
        auto* info = static_cast<char*>(malloc(info_size + 1));

        clGetDeviceInfo(device, cl_device_infos[i], info_size, info, nullptr);
        log_info("- %s: %s\n", cl_device_infos_names[i], info);
    }
}

void read_config(const char* input_file, const char*& preferred_platform, cl_uint& array_width, cl_uint& array_height, cl_float& plate_initial_temperature, float& air_temperature, float& point_temperature)
{
    std::ifstream input(input_file);
    std::string line;

    while (std::getline(input, line)) {
        std::stringstream test(line);
        std::string attribute_name;
        std::string attribute_value;

        std::getline(test, attribute_name, ':');
        std::getline(test, attribute_value, ':');

        if (attribute_name == "width") array_width = std::stoi(attribute_value, nullptr);
        else if (attribute_name == "height") array_height = std::stoi(attribute_value, nullptr);
        else if (attribute_name == "platform" && attribute_value == "Intel") preferred_platform = INTEL_PLATFORM;
        else if (attribute_name == "platform" && attribute_value == "AMD") preferred_platform = AMD_PLATFORM;
        else if (attribute_name == "initial_temp") plate_initial_temperature = std::stof(attribute_value, nullptr);
        else if (attribute_name == "air_temp") air_temperature = std::stof(attribute_value, nullptr);
        else if (attribute_name == "point_temp") point_temperature = std::stof(attribute_value, nullptr);
    }
}

#pragma warning( pop )
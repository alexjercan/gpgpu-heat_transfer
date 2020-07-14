#include "ocl_args.h"

#include <stdlib.h>
#include <tchar.h>
#include <vector>

#include "CL/cl.h"

#include <Windows.h>


#include "log_utils.h"
#include "ocl_context.h"
#include "ocl_kernel.h"
#include "ocl_memory.h"

int _tmain(int argc, TCHAR* argv[])
{
    ocl_args_d_t ocl;
    const cl_device_type device_type = CL_DEVICE_TYPE_GPU;

    const cl_uint array_width = 16;
    const cl_uint array_height = 16;

    auto air_temperature = 10.0F;
    auto plate_temperature = 10.0F;

    auto point_temperature = 100.0F;
    auto point_x = array_width / 2;
    auto point_y = array_height / 2;
    auto point_steps = 10u;

    if (CL_SUCCESS != setup_open_cl(&ocl, device_type))
	    return -1;

    const auto optimized_size = ((sizeof(cl_float) * array_width * array_height - 1) / 64 + 1) * 64;
    auto* input = static_cast<cl_float*>(_aligned_malloc(optimized_size, 4096));
    auto* output = static_cast<cl_float*>(_aligned_malloc(optimized_size, 4096));
    if (nullptr == input || nullptr == output)
    {
        log_error("Error: _aligned_malloc failed to allocate buffers.\n");
        return -1;
    }

    generate_input(input, array_width, array_height, plate_temperature, point_x, point_y, point_temperature);

    if (CL_SUCCESS != create_buffer_arguments(&ocl, input, output, array_width, array_height))
	    return -1;

    if (CL_SUCCESS != setup_ocl_kernel(&ocl, "simulation.cl", "simulate"))
	    return -1;

    do {
        auto axis = 'x';
        if (CL_SUCCESS != set_kernel_arguments(&ocl, array_width, array_height, air_temperature, axis))
	        return -1;
        if (CL_SUCCESS != execute_add_kernel(&ocl, array_width, array_height))
	        return -1;

        auto* aux = ocl.output;
        ocl.output = ocl.input;
        ocl.input = aux;

        axis = 'y';
        if (CL_SUCCESS != set_kernel_arguments(&ocl, array_width, array_height, air_temperature, axis))
            return -1;
        if (CL_SUCCESS != execute_add_kernel(&ocl, array_width, array_height))
            return -1;
    } while (false == read_and_verify(&ocl, array_width, array_height));

    _aligned_free(input);
    _aligned_free(output);

    getchar();

    return 0;
}

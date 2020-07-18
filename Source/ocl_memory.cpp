#include "ocl_memory.h"

#include <cmath>


#include "log_utils.h"
#include "ocl_args.h"


void generate_input(cl_float* input_array, const cl_uint array_width, const cl_uint array_height, const cl_float temperature)
{
    const auto array_size = array_width * array_height;
    for (cl_uint i = 0; i < array_size; ++i)
		input_array[i] = temperature;
}

int create_buffer_arguments(ocl_args_d_t* ocl, cl_float* input, struct vertex_args* plate_points, const cl_uint array_width, const cl_uint array_height)
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

    ocl->input = clCreateImage(ocl->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, &format, &desc, input, &err);
    if (CL_SUCCESS != err)
    {
        log_error("Error: clCreateImage for input returned %s\n", translate_open_cl_error(err));
        return err;
    }

    ocl->output = clCreateImage(ocl->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, &format, &desc, input, &err);
    if (CL_SUCCESS != err)
    {
        log_error("Error: clCreateImage for output returned %s\n", translate_open_cl_error(err));
        return err;
    }

    ocl->plate_points = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, array_width * array_height * sizeof(struct vertex_args), plate_points, &err);
    if (CL_SUCCESS != err)
    {
        log_error("Error: clCreateBuffer for output returned %s\n", translate_open_cl_error(err));
        return err;
    }


    return CL_SUCCESS;
}

bool read_and_verify(ocl_args_d_t* ocl, const cl_uint width, const cl_uint height, struct vertex_args plate_points[])
{
    auto err = CL_SUCCESS;
    auto result = true;

    size_t origin[] = { 0, 0, 0 };
    size_t region[] = { width, height, 1 };
    size_t image_row_pitch;
    size_t image_slice_pitch;
    auto* const result_ptr = static_cast<cl_float*>(clEnqueueMapImage(ocl->command_queue, ocl->output, true, CL_MAP_READ, origin, region,
        &image_row_pitch, &image_slice_pitch, 0, nullptr, nullptr, &err));

    if (CL_SUCCESS != err)
    {
        log_error("Error: clEnqueueMapBuffer returned %s\n", translate_open_cl_error(err));
        return false;
    }
	
    auto* const input_ptr = static_cast<cl_float*>(clEnqueueMapImage(ocl->command_queue, ocl->input, true, CL_MAP_READ, origin, region,
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

    for (unsigned int k = 0; k < size; k++)
    {    		
		if (result && abs(input_ptr[k] - result_ptr[k]) >= CL_FLT_EPSILON * 1000)
		    result = false;
    }

    err = clEnqueueUnmapMemObject(ocl->command_queue, ocl->output, result_ptr, 0, nullptr, nullptr);
    if (CL_SUCCESS != err)
    {
        log_error("Error: clEnqueueUnmapMemObject() returned %s\n", translate_open_cl_error(err));
        return false;
    }

    err = clEnqueueUnmapMemObject(ocl->command_queue, ocl->input, input_ptr, 0, nullptr, nullptr);
    if (CL_SUCCESS != err)
    {
        log_error("Error: clEnqueueUnmapMemObject() returned %s\n", translate_open_cl_error(err));
        return false;
    }

    return result;
}



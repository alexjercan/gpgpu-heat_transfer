#include "ocl_args.h"

#include <stdlib.h>
#include <tchar.h>
#include <vector>

#include "CL/cl.h"

#include <Windows.h>
#include <GLFW/glfw3.h>

#include "log_utils.h"
#include "ocl_context.h"
#include "ocl_kernel.h"
#include "ocl_memory.h"

int setup_ocl(ocl_args_d_t* ocl, const cl_device_type device_type, const char* program_name, const char* kernel_name)
{
	if (CL_SUCCESS != setup_open_cl(ocl, device_type))
		return -1;

	if (CL_SUCCESS != setup_ocl_kernel(ocl, program_name, kernel_name))
		return -1;

    return CL_SUCCESS;
}

void log_device_info(const ocl_args_d_t& ocl)
{
	size_t info_size = 0;
	const cl_device_info cl_device_infos[] = {
		CL_DEVICE_NAME,		CL_DEVICE_VENDOR,
		CL_DEVICE_VERSION,	CL_DRIVER_VERSION,
		CL_DEVICE_PROFILE,	CL_DEVICE_EXTENSIONS };
	const char* cl_device_infos_names[] = {"Name", "Vendor", "Version", "Driver version", "Profile", "Extensions" };
	for (auto i = 0; i < 4; ++i)
	{
		clGetDeviceInfo(ocl.device, cl_device_infos[i], 0, nullptr, &info_size);
		auto* info = static_cast<char*>(malloc(info_size + 1));

		clGetDeviceInfo(ocl.device, cl_device_infos[i], info_size, info, nullptr);
		log_info("- %s: %s\n", cl_device_infos_names[i], info);
	}
}

int setup_device_memory(ocl_args_d_t* ocl, const cl_uint array_width, const cl_uint array_height, const float plate_initial_temperature)
{
	const auto optimized_size = ((sizeof(cl_float) * array_width * array_height - 1) / 64 + 1) * 64;
	auto* input = static_cast<cl_float*>(_aligned_malloc(optimized_size, 4096));
	if (nullptr == input)
	{
		log_error("Error: _aligned_malloc failed to allocate buffers.\n");
		return -1;
	}

	generate_input(input, array_width, array_height, plate_initial_temperature);

	if (CL_SUCCESS != create_buffer_arguments(ocl, input, array_width, array_height))
		return -1;

	_aligned_free(input);

    return CL_SUCCESS;
}

int execute_kernel(ocl_args_d_t& ocl, const cl_uint array_width, const cl_uint array_height, float air_temperature, float point_temperature, unsigned point_x, unsigned point_y)
{
	auto* aux = ocl.output;
	ocl.output = ocl.input;
	ocl.input = aux;
	
	auto axis = 'x';
	if (CL_SUCCESS != set_kernel_arguments(&ocl, array_width, array_height, air_temperature, point_x, point_y, point_temperature, axis))
		return -1;
	if (CL_SUCCESS != execute_add_kernel(&ocl, array_width, array_height))
		return -1;

	aux = ocl.output;
	ocl.output = ocl.input;
	ocl.input = aux;

	axis = 'y';
	if (CL_SUCCESS != set_kernel_arguments(&ocl, array_width, array_height, air_temperature, point_x, point_y, point_temperature, axis))
		return -1;
	if (CL_SUCCESS != execute_add_kernel(&ocl, array_width, array_height))
		return -1;

	return CL_SUCCESS;
}

int main()
{
	/*openCL parameters*/
	ocl_args_d_t ocl;
	const cl_device_type device_type = CL_DEVICE_TYPE_GPU;
	const auto* program_name = "simulation.cl";
	const auto* kernel_name = "simulate";

	/*constant parameters*/
	const cl_uint array_width = 16;
	const cl_uint array_height = 16;
	const auto plate_initial_temperature = 10.0F;

	/*variable parameters*/
	auto air_temperature = 10.0F;
	auto point_temperature = 1500.0F;
	auto point_x = array_width / 2;
	auto point_y = array_height / 2;
	auto steps = 10;
	auto simulate_ocl = true;

	/*setup openCL kernel*/
	if (CL_SUCCESS != setup_ocl(&ocl, device_type, program_name, kernel_name))
		return -1;
	
	/* Initialize the openGL library */
    if (!glfwInit())
	    return -1;

	/*show device info*/
	log_device_info(ocl);

    /* Create a windowed mode window and its OpenGL context */
	GLFWwindow* window = glfwCreateWindow(array_width, array_height, "Heat Transfer Simulation", nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

	/*setup device global memory*/
	if (CL_SUCCESS != setup_device_memory(&ocl, array_width, array_height, plate_initial_temperature))
		return -1;

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
    	/*simulate if the mouse is in the window or if there is not an equilibrium*/
		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);
		point_x = xpos;
		point_y = ypos;
		if (point_x >= 0 && point_x <= array_width && point_y >= 0 && point_y <= array_height)
			simulate_ocl = true;

		/*kernel execution: only if there is not an equilibrium*/
		if (simulate_ocl && CL_SUCCESS != execute_kernel(ocl, array_width, array_height, air_temperature, point_temperature, point_x, point_y))
			return -1;

		/* Render here */
		glClear(GL_COLOR_BUFFER_BIT);

    	/*read temperatures*/
		simulate_ocl = !read_and_verify(&ocl, array_width, array_height);

		/*draw pixels*/
    	
    	
        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

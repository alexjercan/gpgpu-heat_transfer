#include <iosfwd>
#include <iostream>
#include <ostream>
#include <sstream>



#include "ocl_args.h"

#include <stdlib.h>
#include <tchar.h>
#include <vector>

#include "CL/cl.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Windows.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "log_utils.h"
#include "ocl_context.h"
#include "ocl_kernel.h"
#include "ocl_memory.h"
#include "utils.h"

#define APP_NAME "Heat Transfer Simulation"

static const char* vertex_shader_text =
"#version 110\n"
"uniform mat4 MVP;\n"
"attribute vec3 vCol;\n"
"attribute vec2 vPos;\n"
"varying vec3 color;\n"
"void main()\n"
"{\n"
"    gl_Position = MVP * vec4(vPos, 0.0, 1.0);\n"
"    color = vCol;\n"
"}\n";

static const char* fragment_shader_text =
"#version 110\n"
"varying vec3 color;\n"
"void main()\n"
"{\n"
"    gl_FragColor = vec4(color, 1.0);\n"
"}\n";

int setup_ocl(ocl_args_d_t* ocl, const cl_device_type device_type, const char* program_name, const char* kernel_name, const char* preferred_platform)
{
	if (CL_SUCCESS != setup_open_cl(ocl, device_type, preferred_platform))
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

int setup_device_memory(ocl_args_d_t* ocl, struct vertex_args* plate_points, const cl_uint array_width, const cl_uint array_height, const float plate_initial_temperature)
{
	const auto optimized_size = ((sizeof(cl_float) * array_width * array_height - 1) / 64 + 1) * 64;
	auto* input = static_cast<cl_float*>(_aligned_malloc(optimized_size, 4096));
	if (nullptr == input)
	{
		log_error("Error: _aligned_malloc failed to allocate buffers.\n");
		return -1;
	}

	generate_input(input, array_width, array_height, plate_initial_temperature);

	if (CL_SUCCESS != create_buffer_arguments(ocl, input, plate_points, array_width, array_height))
		return -1;

	_aligned_free(input);

    return CL_SUCCESS;
}

int execute_kernel(ocl_args_d_t& ocl, const cl_uint array_width, const cl_uint array_height, float air_temperature, float point_temperature, unsigned point_x, unsigned point_y)
{
	auto* aux = ocl.output;
	ocl.output = ocl.input;
	ocl.input = aux;
	
	if (CL_SUCCESS != set_kernel_arguments(&ocl, array_width, array_height, air_temperature, point_x, point_y, point_temperature))
		return -1;
	if (CL_SUCCESS != execute_add_kernel(&ocl, array_width, array_height))
		return -1;
	
	return CL_SUCCESS;
}

void gl_setup_shader(GLuint& program, GLint& mvp_location)
{
	/*setup openGL shader*/
	auto vertex_shader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex_shader, 1, &vertex_shader_text, nullptr);
	glCompileShader(vertex_shader);

	auto fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment_shader, 1, &fragment_shader_text, nullptr);
	glCompileShader(fragment_shader);

	program = glCreateProgram();
	glAttachShader(program, vertex_shader);
	glAttachShader(program, fragment_shader);
	glLinkProgram(program);

	mvp_location = glGetUniformLocation(program, "MVP");
	const auto vpos_location = glGetAttribLocation(program, "vPos");
	const auto vcol_location = glGetAttribLocation(program, "vCol");

	glEnableVertexAttribArray(vpos_location);
	glVertexAttribPointer(vpos_location, 2, GL_FLOAT, GL_FALSE, sizeof(struct vertex_args), static_cast<void*>(nullptr));
	glEnableVertexAttribArray(vcol_location);
	glVertexAttribPointer(vcol_location, 3, GL_FLOAT, GL_FALSE, sizeof(struct vertex_args), reinterpret_cast<void*>(sizeof(float) * 2));
}

void draw_pixels(cl_uint array_width, cl_uint array_height, vertex_args** plate_points, GLFWwindow* window, GLuint vertex_buffer, GLuint program, GLint mvp_location)
{
	/*setup viewport*/
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	glViewport(0, 0, width, height);
    	
	/*setup camera*/
	glm::mat4x4 m(1.0F);
	auto p = glm::ortho(-1.0F, 1.0F, 1.0F, -1.0F);
	auto mvp = p * m;

	/*draw pixels*/
	glUnmapBuffer(GL_ARRAY_BUFFER);
	glUseProgram(program);
	glUniformMatrix4fv(mvp_location, 1, GL_FALSE, glm::value_ptr(mvp));
	glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
	glDrawArrays(GL_POINTS, 0, array_width * array_height);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
	*plate_points = static_cast<struct vertex_args*>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE));
}

void calculate_mouse_position(cl_uint array_width, cl_uint array_height, int& point_x, int& point_y, bool& simulate_ocl, GLFWwindow* window)
{
	/*simulate if the mouse is in the window or if there is not an equilibrium*/
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);
	point_x = xpos;
	point_y = ypos;
	if (point_x >= 0 && point_x <= array_width && point_y >= 0 && point_y <= array_height)
		simulate_ocl = true;
}

void create_gl_buffer(cl_uint array_width, cl_uint array_height, GLuint& vertex_buffer, struct vertex_args** plate_points)
{
	glGenBuffers(1, &vertex_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
	glBufferData(GL_ARRAY_BUFFER, array_width * array_height * sizeof(struct vertex_args), nullptr, GL_DYNAMIC_DRAW);
	*plate_points = static_cast<struct vertex_args*>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE));
	for (cl_uint i = 0; i < array_height; i++)
	{
		for (cl_uint j = 0; j < array_width; j++)
		{
			static_cast<struct vertex_args*>(*plate_points)[i * array_width + j].x = 2 * (j / static_cast<float>(array_width) - 0.5F);
			static_cast<struct vertex_args*>(*plate_points)[i * array_width + j].y = 2 * (i / static_cast<float>(array_height) - 0.5F);
		}
	}
}

int setup_ogl(cl_uint array_width, cl_uint array_height, GLFWwindow*& window)
{
	/* Initialize the openGL library */
	if (!glfwInit())
		return -1;

	window = glfwCreateWindow(array_width, array_height, APP_NAME, nullptr, nullptr);
	if (!window)
	{
		glfwTerminate();
		return -1;
	}

	/* Make the window's context current */
	glfwMakeContextCurrent(window);

	if (GLEW_OK != glewInit())
		return -1;

	return 0;
}

void show_fps(GLFWwindow* pWindow, float fps)
{
	std::stringstream ss;
	ss << APP_NAME << " [" << fps << " FPS]";

	glfwSetWindowTitle(pWindow, ss.str().c_str());
}

int main()
{
	ocl_args_d_t ocl;
	const cl_device_type device_type = CL_DEVICE_TYPE_GPU;
	const char* preferred_platform = INTEL_PLATFORM;
	const auto* program_name = "simulation.cl";
	const auto* kernel_name = "simulate";
	const auto* input_file = "config.in";
	cl_uint array_width = 640;
	cl_uint array_height = 480;
	cl_float plate_initial_temperature = 10.0F;
	auto air_temperature = 100.0F;
	auto point_temperature = 1500.0F;
	auto point_x = 0;
	auto point_y = 0;
	auto simulate_ocl = true;
	struct vertex_args* plate_points = nullptr;
	
	LARGE_INTEGER perf_frequency;
	LARGE_INTEGER performance_count_nd_range_start;
	LARGE_INTEGER performance_count_nd_range_stop;

	read_config(input_file, preferred_platform, array_width, array_height, plate_initial_temperature, air_temperature, point_temperature);

	/*setup openCL kernel*/
	if (CL_SUCCESS != setup_ocl(&ocl, device_type, program_name, kernel_name, preferred_platform))
		return -1;

	/*show device info*/
	log_device_info(ocl);
	/*show simulation info*/
	log_info("\nwidth=%u\nheight=%u\nplate_temp=%f\nair_temp=%f\npoint_temp=%f\n", array_width, array_height, plate_initial_temperature, air_temperature, point_temperature);

	/*setup openGL*/
	GLFWwindow* window;
	if (0 != setup_ogl(array_width, array_height, window)) return -1;

	/*create vertex buffer*/
	GLuint vertex_buffer;
	create_gl_buffer(array_width, array_height, vertex_buffer, &plate_points);

	/*initialize shader*/
	GLuint program;
	GLint mvp_location;
	gl_setup_shader(program, mvp_location);
	
	/*setup device global memory*/
	if (CL_SUCCESS != setup_device_memory(&ocl, plate_points, array_width, array_height, plate_initial_temperature))
		return -1;

	/* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
		QueryPerformanceCounter(&performance_count_nd_range_start);

    	/*input*/
    	calculate_mouse_position(array_width, array_height, point_x, point_y, simulate_ocl, window);

		/*kernel execution: only if there is not an equilibrium*/
		if (simulate_ocl && CL_SUCCESS != execute_kernel(ocl, array_width, array_height, air_temperature, point_temperature, point_x, point_y))
			return -1;

		/* Render here */
		glClear(GL_COLOR_BUFFER_BIT);

    	/*read temperatures and update the plate points*/
		simulate_ocl = !read_and_verify(&ocl, array_width, array_height, plate_points);
		if (simulate_ocl == false) log_info("Convergence reached.\n");

		draw_pixels(array_width, array_height, &plate_points, window, vertex_buffer, program, mvp_location);
    	
        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();

    	/*performance*/
		QueryPerformanceCounter(&performance_count_nd_range_stop);
		QueryPerformanceFrequency(&perf_frequency);
		float mstime = 1000.0f * static_cast<float>(performance_count_nd_range_stop.QuadPart - performance_count_nd_range_start.QuadPart) / static_cast<float>(perf_frequency.QuadPart);
		float fps = 1000.0f / mstime;
		show_fps(window, fps);
    }

	glfwTerminate();
    return 0;
}

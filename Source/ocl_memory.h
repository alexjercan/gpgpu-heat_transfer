#pragma once
#include <CL/cl.h>

struct vertex_args
{
	float x, y;
	float r, g, b;
};

#define TEMPERATURES_COUNT 11

const struct vertex_args temperature_color[TEMPERATURES_COUNT] = {
	{0.0F, 200.0F, 0.05F, 0.05F, 0.05F}, //normal
	{200.0F, 426.0F, 0.08F, 0.05F, 0.05F}, //heated
	{426.0F, 593.0F, 0.14F, 0.05F, 0.05F}, //black red
	{593.0F, 704.0F, 0.24F, 0.05F, 0.06F}, //very dark red
	{704.0F, 814.0F, 0.34F, 0.07F, 0.06F}, //dark red
	{815.0F, 870.0F, 0.54F, 0.08F, 0.07F}, //cherry red
	{871.0F, 981.0F, 0.74F, 0.2F, 0.07F}, //light cherry red
	{981.0F, 1092.0F, 0.84F, 0.5F, 0.08F}, //Orange
	{1093.0F, 1258.0F, 1.0F, 1.0F, 0.1F}, //Yellow
	{1259.0F, 1314.0F, 1.0F, 1.0F, 0.8F}, //Yellow white
	{1315.0F, -1.0F, 1.0F, 1.0F, 1.0F}, //White
};

struct ocl_args_d_t;

void generate_input(cl_float* input_array, cl_uint array_width, cl_uint array_height, cl_float temperature);
int create_buffer_arguments(ocl_args_d_t* ocl, cl_float* input, struct vertex_args* plate_points, const cl_uint array_width, const cl_uint array_height);
bool read_and_verify(ocl_args_d_t* ocl, const cl_uint width, const cl_uint height, struct vertex_args plate_points[]);

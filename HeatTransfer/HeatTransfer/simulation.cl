constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
constant float gaussian_kernel = 1/9.0F;

#define TEMPERATURES_COUNT 11

struct vertex_args
{
	float x, y;
	float r, g, b;
};

__constant struct vertex_args temperature_color[TEMPERATURES_COUNT] = {
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

__kernel void simulate(read_only image2d_t input, write_only image2d_t output, uint width, uint height, float air_temperature, uint point_x, uint point_y, float point_temperature, __global struct vertex_args* plate_points, float gpu_percenet)
{
    int2 coords = (int2)(get_global_id(0), get_global_id(1));
	int global_index = coords.y * width + coords.x;
	float4 color = (float4)(0.0F, 0.0F, 0.0F, 0.0F);
	float4 ext_color = (float4)(air_temperature, air_temperature, air_temperature, air_temperature);
	int i, j;

	if (global_index > width * height * gpu_percenet / 100.0F)
	{
		return;
	}

	if (point_x == coords.x && point_y == coords.y)
	{
		color = (float4)(point_temperature, point_temperature, point_temperature, point_temperature);
		write_imagef(output, coords, color);
		return;
	}

	for (i = -1; i <= 1; i++)
	{
		for (j = -1; j <= 1; j++)
		{
			if (coords.x + i < 0 || coords.x + i >= width || coords.y + i < 0 || coords.y + i >= height) color += ext_color * gaussian_kernel;
			else color += read_imagef(input, sampler, (int2)(coords.x + i, coords.y + j)) * gaussian_kernel;
		}
	}

	if (color.x < temperature_color[9].x)
    {
        for (int i = 0; i < TEMPERATURES_COUNT - 1; i++)
        {
            if (color.x < temperature_color[i].y)
            {
                float diff = color.x - temperature_color[i].x;
                float diff_total = temperature_color[i].y - temperature_color[i].x;
                float proc = diff / diff_total;

                plate_points[global_index].r = temperature_color[i].r + (temperature_color[i + 1].r - temperature_color[i].r) * proc;
                plate_points[global_index].g = temperature_color[i].g + (temperature_color[i + 1].g - temperature_color[i].g) * proc;
                plate_points[global_index].b = temperature_color[i].b + (temperature_color[i + 1].b - temperature_color[i].b) * proc;
                break;
            }
        }
    }
    else
    {
        plate_points[global_index].r = temperature_color[TEMPERATURES_COUNT - 1].r;
        plate_points[global_index].g = temperature_color[TEMPERATURES_COUNT - 1].g;
        plate_points[global_index].b = temperature_color[TEMPERATURES_COUNT - 1].b;
    }

    write_imagef(output, coords, color);
}
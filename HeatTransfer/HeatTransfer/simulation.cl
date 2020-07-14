constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
constant float gaussian_kernel = 1/3.0F;

__kernel void simulate(read_only image2d_t input, write_only image2d_t output, uint width, uint height, float temperature, uint point_x, uint point_y, float point_temperature, char axis)
{
    int2 coords = (int2)(get_global_id(0), get_global_id(1));
	//int global_index = coords.y * width + coords.x;
	float4 color = (float4)(0.0F, 0.0F, 0.0F, 0.0F);
	float4 ext_color = (float4)(temperature, temperature, temperature, temperature);
	int i;

	if (point_x == coords.x && point_y == coords.y)
	{
		color = (float4)(point_temperature, point_temperature, point_temperature, point_temperature);
		write_imagef(output, coords, color);
		return;
	}

	for (i = -1; i <= 1; i++)
	{
		switch (axis)
		{
			case 'x':
				if (coords.x + i < 0 || coords.x + i >= width) color += ext_color * gaussian_kernel;
				else color += read_imagef(input, sampler, (int2)(coords.x + i, coords.y)) * gaussian_kernel;
				break;

			case 'y':
				if (coords.y + i < 0 || coords.y + i >= height) color += ext_color * gaussian_kernel;
				else color += read_imagef(input, sampler, (int2)(coords.x, coords.y + i)) * gaussian_kernel;
				break;
		}
	}

    write_imagef(output, coords, color);
}
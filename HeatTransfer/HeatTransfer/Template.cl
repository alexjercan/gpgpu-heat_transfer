constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void Add(read_only image2d_t imageA, read_only image2d_t imageB, write_only image2d_t imageC)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    uint A = read_imageui(imageA, sampler, (int2)(x, y)).x;
    uint B = read_imageui(imageB, sampler, (int2)(x, y)).x;

    write_imageui(imageC, (int2)(x, y), A + B);
}

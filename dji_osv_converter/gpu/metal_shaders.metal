// Metal compute shaders for GPU-accelerated video processing
// All shaders operate on GPU textures, no CPU-side conversion

#include <metal_stdlib>
using namespace metal;

// YUV to RGB conversion
// Input: YUV420p texture (Y plane + UV planes)
// Output: RGB texture

kernel void yuv420_to_rgb24(
    texture2d<float, access::read> y_texture [[texture(0)]],
    texture2d<float, access::read> uv_texture [[texture(1)]],
    texture2d<float, access::write> rgb_texture [[texture(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    // Ensure we're within bounds
    if (gid.x >= rgb_texture.get_width() || gid.y >= rgb_texture.get_height()) {
        return;
    }

    // Sample Y plane (1:1)
    float y = y_texture.read(gid).r;

    // Sample UV planes (2:1 in each dimension for 4:2:0)
    uint2 uv_coord = gid / 2;
    float4 uv = uv_texture.read(uv_coord);
    float u = uv.r - 0.5;
    float v = uv.g - 0.5;

    // BT.709 YUV to RGB conversion matrix
    // R = Y + 1.5748 * V
    // G = Y - 0.1873 * U - 0.4681 * V
    // B = Y + 1.8556 * U

    float r = y + 1.5748 * v;
    float g = y - 0.1873 * u - 0.4681 * v;
    float b = y + 1.8556 * u;

    // Clamp to [0, 1] range
    r = clamp(r, 0.0, 1.0);
    g = clamp(g, 0.0, 1.0);
    b = clamp(b, 0.0, 1.0);

    rgb_texture.write(float4(r, g, b, 1.0), gid);
}

// RGB to YUV conversion
// Input: RGB texture
// Output: YUV texture

kernel void rgb24_to_yuv420(
    texture2d<float, access::read> rgb_texture [[texture(0)]],
    texture2d<float, access::write> y_texture [[texture(1)]],
    texture2d<float, access::write> uv_texture [[texture(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    // Ensure we're within bounds
    if (gid.x >= rgb_texture.get_width() || gid.y >= rgb_texture.get_height()) {
        return;
    }

    float4 rgb = rgb_texture.read(gid);
    float r = rgb.r;
    float g = rgb.g;
    float b = rgb.b;

    // BT.709 RGB to YUV conversion matrix
    // Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    // U = -0.1146 * R - 0.3854 * G + 0.5 * B + 0.5
    // V = 0.5 * R - 0.4542 * G - 0.0458 * B + 0.5

    float y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    float u = -0.1146 * r - 0.3854 * g + 0.5 * b + 0.5;
    float v = 0.5 * r - 0.4542 * g - 0.0458 * b + 0.5;

    // Write Y plane
    y_texture.write(float4(y, 0.0, 0.0, 1.0), gid);

    // Write UV planes (only for even coordinates to maintain 4:2:0)
    if ((gid.x % 2 == 0) && (gid.y % 2 == 0)) {
        uint2 uv_coord = gid / 2;
        uv_texture.write(float4(u, v, 0.0, 1.0), uv_coord);
    }
}

// Fisheye undistortion using lookup table
// Input: fisheye RGB texture, lookup map
// Output: undistorted RGB texture

kernel void fisheye_undistort(
    texture2d<float, access::read> input_texture [[texture(0)]],
    texture2d<float, access::read> map_x [[texture(1)]],
    texture2d<float, access::read> map_y [[texture(2)]],
    texture2d<float, access::write> output_texture [[texture(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= output_texture.get_width() || gid.y >= output_texture.get_height()) {
        return;
    }

    // Read the source coordinates from the lookup map
    float src_x = map_x.read(gid).r;
    float src_y = map_y.read(gid).r;

    // Bilinear interpolation from source
    uint2 src_coord = uint2(src_x, src_y);
    float4 pixel = input_texture.read(src_coord);

    output_texture.write(pixel, gid);
}

// Seam blending for frame stitching
// Input: two frames, overlap region mask
// Output: blended frame

kernel void seam_blend(
    texture2d<float, access::read> left_texture [[texture(0)]],
    texture2d<float, access::read> right_texture [[texture(1)]],
    texture2d<float, access::read> blend_mask [[texture(2)]],
    texture2d<float, access::write> output_texture [[texture(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= output_texture.get_width() || gid.y >= output_texture.get_height()) {
        return;
    }

    float4 left = left_texture.read(gid);
    float4 right = right_texture.read(gid);
    float alpha = blend_mask.read(gid).r;

    // Linear blend
    float4 blended = mix(left, right, alpha);

    output_texture.write(blended, gid);
}

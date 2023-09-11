const std = @import("std");
const data_point = @import("datapoint.zig");

pub fn free_mnist_data_points_soa(allocator: std.mem.Allocator, soa: data_point.DataPointSOA) void {
    allocator.free(soa.x);
    allocator.free(soa.y);
}

/// Assumed to be the same length
fn copy_image_data(input: []const u8, output: []f64) void {
    for (input) |pixel, i| {
        const x_val = @intToFloat(f64, pixel);
        // normalize
        output[i] = x_val / 255;
    }
}

/// Need to call `free_mnist_data_points_soa`
pub fn make_mnist_data_points_soa(allocator: std.mem.Allocator, x: []const u8, x_chunk_size: usize, y: []const u8, y_output_size: usize) !data_point.DataPointSOA {
    var x_data = try allocator.alloc(f64, x.len);
    var y_data = try allocator.alloc(f64, y.len * y_output_size);
    const result = data_point.DataPointSOA{
        .x = x_data,
        .x_chunk_size = x_chunk_size,
        .y_chunk_size = y_output_size,
        .y = y_data,
    };
    var idx: usize = 0;
    const total = x.len / x_chunk_size;
    while (idx < total) : (idx += 1) {
        const i = idx * x_chunk_size;
        const x_chunk = x[i .. i + x_chunk_size];
        const x_buffer = result.x[i .. i + x_chunk_size];
        copy_image_data(x_chunk, x_buffer);
        const y_index = idx * result.y_chunk_size;
        const y_buffer = result.y[y_index .. y_index + result.y_chunk_size];
        write_digit(y[idx], y_buffer);
    }
    return result;
}

/// Writes the decimal digit 0-9 to a buffer
/// of size 10, where the value at the position
/// corresponds to whether the current digit
/// is represented.
///
/// For example, `4` is represented as
///
/// ```
/// [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
///  0  1  2  3  4  5  6  7  8  9
/// ```
fn write_digit(digit: u8, buf: []f64) void {
    // clear all
    for (buf) |_, i| {
        buf[i] = 0;
    }
    buf[digit] = 1;
}

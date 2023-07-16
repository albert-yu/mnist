const std = @import("std");
const lin = @import("linalg.zig");
const nn = @import("network.zig");

/// https://stackoverflow.com/a/73020142
fn range(len: usize) []const void {
    return @as([*]void, undefined)[0..len];
}

fn get_double_word(bytes: []u8, offset: usize) u32 {
    const slice = bytes[offset .. offset + 4][0..4];
    return std.mem.readInt(u32, slice, std.builtin.Endian.Big);
}

fn console_print_image(img_bytes: []u8, num_rows: usize) void {
    // console print
    for (img_bytes) |pixel, i| {
        if (i % num_rows == 0) {
            std.debug.print("\n", .{});
        }
        if (pixel == 0) {
            std.debug.print("0", .{});
        } else {
            std.debug.print("1", .{});
        }
    }
    std.debug.print("\n", .{});
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
fn write_digit(digit: u8, buf: *[10]f32) void {
    // clear all
    for (buf) |_, i| {
        buf[i] = 0;
    }
    buf[digit] = 1;
}

/// Assumed to be the same length
fn copy_image_data(input: []u8, output: []f32) void {
    for (input) |pixel, i| {
        output[i] = @intToFloat(f32, pixel);
    }
}

fn read_file(allocator: std.mem.Allocator, filename: []const u8) !void {
    const file = try std.fs.cwd().openFile(filename);
    defer file.close();

    const size = try file.getEndPos();
    const buffer = try allocator.alloc(u8, size);
    _ = try file.read(buffer);
    return buffer;
}


pub fn main() !void {
    const TRAIN_LABELS_FILE = "data/train-labels.idx1-ubyte";
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    const train_labels_buffer = try read_file(allocator, TRAIN_LABELS_FILE);
    defer allocator.free(train_labels_buffer);

    // read training labels
    const count_offset = 4;
    const train_count = get_double_word(train_labels_buffer, count_offset);

    std.debug.print("label count: {}\n", .{train_count});
    const start_index = 8;
    const labels = train_labels_buffer[start_index..];

    // read training images
    const TRAIN_IMAGES_FILE = "data/train-images.idx3-ubyte";
    const train_images_buffer = try read_file(allocator, TRAIN_IMAGES_FILE);
    defer allocator.free(train_images_buffer);

    // can read image count from file, but should be exactly the same as labels
    const num_rows = get_double_word(train_images_buffer, 8);
    const num_cols = get_double_word(train_images_buffer, 12);
    std.debug.print("rows: {}, cols: {}\n", .{ num_rows, num_cols });
    const img_start_offset = 16;
    const images = train_images_buffer[img_start_offset..];

    const image_size = num_rows * num_cols;

}

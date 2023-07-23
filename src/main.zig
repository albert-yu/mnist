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

fn read_file(allocator: std.mem.Allocator, filename: []const u8) ![]u8 {
    const file = try std.fs.cwd().openFile(filename, .{});
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

    const DIGITS = 10;
    const train_data_points = try nn.make_mnist_data_points(allocator, images, image_size, labels, DIGITS);
    defer nn.free_mnist_data_points(allocator, train_data_points);

    const HIDDEN_LAYER_SIZE = 30;
    const layer_sizes = [_]usize{ image_size, HIDDEN_LAYER_SIZE, DIGITS };
    var network = try nn.alloc_network(allocator, &layer_sizes);
    defer nn.free_network(allocator, network);
    try network.sgd(train_data_points, 0.05);
}

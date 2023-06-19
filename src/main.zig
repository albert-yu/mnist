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

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    // defer allocator.deinit();
    const train_labels_file = try std.fs.cwd().openFile("data/train-labels.idx1-ubyte", .{});
    defer train_labels_file.close();

    const size = try train_labels_file.getEndPos();
    const train_labels_buffer = try allocator.alloc(u8, size);
    defer allocator.free(train_labels_buffer);
    _ = try train_labels_file.read(train_labels_buffer);

    // read training labels
    const count_offset = 4;
    const train_count = get_double_word(train_labels_buffer, count_offset);

    std.debug.print("label count: {}\n", .{train_count});
    const start_index = 8;
    const labels = train_labels_buffer[start_index..];

    // read training images
    const file_train_img = try std.fs.cwd().openFile("data/train-images.idx3-ubyte", .{});
    defer file_train_img.close();

    const train_im_file_size = try file_train_img.getEndPos();
    const train_images_buffer = try allocator.alloc(u8, train_im_file_size);
    defer allocator.free(train_images_buffer);
    _ = try file_train_img.read(train_images_buffer);

    // can read image count from file, but should be exactly the same as labels
    const num_rows = get_double_word(train_images_buffer, 8);
    const num_cols = get_double_word(train_images_buffer, 12);
    std.debug.print("rows: {}, cols: {}\n", .{ num_rows, num_cols });
    const img_start_offset = 16;
    const images = train_images_buffer[img_start_offset..];

    const image_size = num_rows * num_cols;

    // construct the network
    const POSSIBLE_DIGITS = 10;

    const HIDDEN_LAYER_SIZE = 15;
    // init hidden layer
    var hidden_layer_matrix_buf = try allocator.alloc(f32, HIDDEN_LAYER_SIZE * image_size);
    defer allocator.free(hidden_layer_matrix_buf);

    var hidden_layer_biases = [_]f32{0} ** HIDDEN_LAYER_SIZE;
    var hidden_layer_z_vector = [_]f32{0} ** HIDDEN_LAYER_SIZE;
    var hidden_layer_activations = [_]f32{0} ** HIDDEN_LAYER_SIZE;
    var hidden_layer = nn.NetworkLayer{
        .weights = lin.Matrix{
            .data = hidden_layer_matrix_buf,
            .rows = HIDDEN_LAYER_SIZE,
            .cols = image_size,
        },
        .biases = &hidden_layer_biases,
        .z_vector = &hidden_layer_z_vector,
        .activations = &hidden_layer_activations,
    };

    comptime var OUTPUT_LAYER_MATRIX_SIZE = POSSIBLE_DIGITS * HIDDEN_LAYER_SIZE;
    var output_layer_buf = try allocator.alloc(f32, OUTPUT_LAYER_MATRIX_SIZE);
    defer allocator.free(output_layer_buf);

    var out_layer_biases = [_]f32{0} ** POSSIBLE_DIGITS;
    var out_layer_z_vector = [_]f32{0} ** POSSIBLE_DIGITS;
    var out_layer_activations = [_]f32{0} ** POSSIBLE_DIGITS;
    var output_layer = nn.NetworkLayer{
        .weights = lin.Matrix{
            .data = output_layer_buf,
            .rows = POSSIBLE_DIGITS,
            .cols = HIDDEN_LAYER_SIZE,
        },
        .biases = &out_layer_biases,
        .z_vector = &out_layer_z_vector,
        .activations = &out_layer_activations,
    };

    // put layers together
    var all_layers = [_]nn.NetworkLayer{
        hidden_layer,
        output_layer,
    };
    var mnist_network = nn.Network{
        .layers = &all_layers,
    };

    var biases_1 = try allocator.alloc(f32, HIDDEN_LAYER_SIZE);
    var weights_1_data = try allocator.alloc(f32, HIDDEN_LAYER_SIZE * image_size);
    var result_placeholder_1 = nn.GradientResult{
        .biases = biases_1,
        .weights = lin.Matrix{
            .data = weights_1_data,
            .rows = HIDDEN_LAYER_SIZE,
            .cols = image_size,
        },
    };
    var biases_2 = [_]f32{0} ** POSSIBLE_DIGITS;
    var weights_2_data = [_]f32{0} ** (POSSIBLE_DIGITS * HIDDEN_LAYER_SIZE);
    var result_placeholder_2 = nn.GradientResult{
        .biases = &biases_2,
        .weights = lin.Matrix{
            .data = &weights_2_data,
            .rows = POSSIBLE_DIGITS,
            .cols = HIDDEN_LAYER_SIZE,
        },
    };
    var results = [_]nn.GradientResult{
        result_placeholder_1,
        result_placeholder_2,
    };

    // train the network
    var expected_output_buf = [_]f32{0} ** POSSIBLE_DIGITS;
    var images_as_input = try allocator.alloc(f32, image_size);
    defer allocator.free(images_as_input);
    for (range(train_count)) |_, i| {
        if (i >= 20) {
            break;
        }
        // labeled digit
        const digit = labels[i];
        write_digit(digit, &expected_output_buf);

        // digit image
        const img_slice_start = i * image_size;
        const img_slice = images[img_slice_start..image_size];
        copy_image_data(img_slice, images_as_input);
        try mnist_network.backprop(images_as_input, &expected_output_buf, &results);
    }
}

const std = @import("std");
const linalg = @import("linalg.zig");
const layers = @import("layer.zig");
const mnist = @import("mnist.zig");
const maths = @import("maths.zig");
const perf = @import("performance.zig");

fn get_double_word(bytes: []u8, offset: usize) u32 {
    const slice = bytes[offset .. offset + 4][0..4];
    return std.mem.readInt(u32, slice, std.builtin.Endian.Big);
}

fn console_print_image(img_bytes: []f64, num_rows: usize) void {
    // console print
    for (img_bytes, 0..) |pixel, i| {
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

fn find_max_index(buf: []f64) usize {
    var max_i: usize = 0;
    var max: f64 = buf[0];
    for (buf, 0..) |val, i| {
        if (i == 0) {
            continue;
        }
        if (val > max) {
            max_i = i;
            max = val;
        }
    }
    return max_i;
}

pub fn main() !void {
    const TRAIN_LABELS_FILE = "data/train-labels.idx1-ubyte";
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    const train_labels_buffer = try read_file(allocator, TRAIN_LABELS_FILE);
    defer allocator.free(train_labels_buffer);

    // read training labels
    const start_index = 8;
    const labels = train_labels_buffer[start_index..];

    // read training images
    const TRAIN_IMAGES_FILE = "data/train-images.idx3-ubyte";
    const train_images_buffer = try read_file(allocator, TRAIN_IMAGES_FILE);
    defer allocator.free(train_images_buffer);

    // can read image count from file, but should be exactly the same as labels
    const img_start_offset = 16;
    const images = train_images_buffer[img_start_offset..];

    // read test images
    const TEST_IMAGES_FILE = "data/t10k-images.idx3-ubyte";
    const test_images_buffer = try read_file(allocator, TEST_IMAGES_FILE);
    defer allocator.free(test_images_buffer);
    const test_images = test_images_buffer[img_start_offset..];

    // read test labels
    const TEST_LABELS_FILE = "data/t10k-labels.idx1-ubyte";
    const test_labels_buffer = try read_file(allocator, TEST_LABELS_FILE);
    defer allocator.free(test_labels_buffer);
    const test_labels = test_labels_buffer[start_index..];

    const image_size = 28 * 28;

    const DIGITS = 10;
    std.debug.print("making training data points...", .{});
    const train_data_points = try mnist.make_mnist_data_points_soa(allocator, images, image_size, labels, DIGITS);
    defer mnist.free_mnist_data_points_soa(allocator, train_data_points);
    std.debug.print("made {} train data points.\n", .{train_data_points.len()});

    std.debug.print("making test data points...", .{});
    const test_data = try mnist.make_mnist_data_points_soa(allocator, test_images, image_size, test_labels, DIGITS);
    std.debug.print("made {} test data points.\n", .{test_data.len()});

    const HIDDEN_LAYER_SIZE = 30;
    const ETA = 0.05;
    const EPOCHS = 100;

    var layer1 = try layers.Layer(image_size, HIDDEN_LAYER_SIZE).alloc(allocator);
    defer layer1.dealloc(allocator);
    var layer2 = try layers.Layer(HIDDEN_LAYER_SIZE, DIGITS).alloc(allocator);
    defer layer2.dealloc(allocator);

    layer1.init_randn();
    layer2.init_randn();

    const BATCH_SIZE = 10;

    const batch_count = train_data_points.len() / BATCH_SIZE;

    var stopwatch = perf.Stopwatch.new();
    var epoch_index: usize = 0;
    while (epoch_index < EPOCHS) : (epoch_index += 1) {
        std.debug.print("training epoch {} of {}\n", .{ epoch_index + 1, EPOCHS });
        stopwatch.start();
        try train_data_points.shuffle(allocator);
        stopwatch.report("shuffle");
        var batch_index: usize = 0;
        const scalar = ETA / @as(f64, @floatFromInt(BATCH_SIZE));
        while (batch_index < batch_count) : (batch_index += 1) {
            var i = batch_index * BATCH_SIZE;
            const end = i + BATCH_SIZE;
            while (i < end) : (i += 1) {
                var x_data = train_data_points.x_at(i);
                // console_print_image(x_data, 28);
                var y_data = train_data_points.y_at(i);

                var x = linalg.Matrix{
                    .data = x_data,
                    .rows = x_data.len,
                    .cols = 1,
                };

                var y = linalg.Matrix{
                    .data = y_data,
                    .rows = y_data.len,
                    .cols = 1,
                };

                // forward
                var activations1 = try layer1.forward(allocator, x, maths.sigmoid);
                defer activations1.dealloc_data(allocator);

                var activations2 = try layer2.forward(allocator, activations1, maths.sigmoid);
                defer activations2.dealloc_data(allocator);

                var err = try activations2.sub_alloc(allocator, y);
                defer err.dealloc_data(allocator);

                // backward
                var grad2 = try layer2.backward(allocator, err, maths.sigmoid_prime);
                defer grad2.dealloc(allocator);

                var err_inner = try layer2.weights.t_alloc(allocator);
                defer err_inner.dealloc_data(allocator);

                err_inner.multiply(grad2.biases, &err_inner);
                var grad1 = try layer1.backward(allocator, err_inner, maths.sigmoid_prime);
                defer grad1.dealloc(allocator);

                // apply gradients
                grad1.biases.scale(scalar);
                layer1.biases.sub(grad1.biases, &layer1.biases);
                grad1.weights.scale(scalar);
                layer1.weights.sub(grad1.weights, &layer1.weights);

                grad2.biases.scale(scalar);
                layer2.biases.sub(grad2.biases, &layer2.biases);
                grad2.weights.scale(scalar);
                layer2.weights.sub(grad2.weights, &layer2.weights);
            }
        }
        stopwatch.report("trained");
        std.debug.print("evaluating...", .{});

        var i: usize = 0;
        var correct: usize = 0;
        while (i < test_data.len()) : (i += 1) {
            var x_data = test_data.x_at(i);
            var x = linalg.Matrix{
                .data = x_data,
                .rows = image_size,
                .cols = 1,
            };
            var y_data = test_data.y_at(i);

            var activations1 = try layer1.forward(allocator, x, maths.sigmoid);
            defer activations1.dealloc_data(allocator);

            var activations2 = try layer2.forward(allocator, activations1, maths.sigmoid);
            defer activations2.dealloc_data(allocator);

            const expected = find_max_index(y_data);
            const actual = find_max_index(activations2.data);
            if (expected == actual) {
                correct += 1;
            }
        }
        std.debug.print("done. {}/{} correct\n", .{ correct, test_data.len() });
    }

    std.debug.print("done.\n", .{});
}

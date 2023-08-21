const std = @import("std");
const linalg = @import("linalg.zig");
const maths = @import("maths.zig");

fn shallow_copy_slice(comptime T: type, source: []const T, dest: []T) void {
    for (source) |elem, i| {
        dest[i] = elem;
    }
}

/// resulting buffer can be freed with allocator.free()
fn alloc_copy(comptime T: type, allocator: std.mem.Allocator, source: []const T) error{OutOfMemory}![]T {
    var buffer = try allocator.alloc(T, source.len);
    shallow_copy_slice(T, source, buffer);
    return buffer;
}

const randgen = std.rand.DefaultPrng;

fn shuffle(comptime T: type, arr: []T) void {
    var rand = randgen.init(0);
    for (arr) |elem, i| {
        const random_offset = rand.random().int(usize);
        const new_i = (i + random_offset) % arr.len;
        // swap
        const temp = arr[new_i];
        arr[new_i] = elem;
        arr[i] = temp;
    }
}

const FeedforwardResult = struct { activations: []linalg.Matrix, z_results: []linalg.Matrix };

fn free_feedforward(allocator: std.mem.Allocator, result: FeedforwardResult) void {
    for (result.activations) |activation| {
        linalg.free_matrix_data(allocator, activation);
    }
    allocator.free(result.activations);

    for (result.z_results) |z| {
        linalg.free_matrix_data(allocator, z);
    }
    allocator.free(result.z_results);
}

fn delta_to_w(allocator: std.mem.Allocator, delta_ptr: *linalg.Matrix, prev_activation: *linalg.Matrix, nabla_w_ptr: *linalg.Matrix) !void {
    // activations[-l].transpose()
    var prev_activation_transposed = try linalg.alloc_matrix(allocator, prev_activation.cols, prev_activation.rows);
    defer linalg.free_matrix(allocator, prev_activation_transposed);
    linalg.transpose(prev_activation.*, prev_activation_transposed);

    // delta (dot) activations[-2].transpose()
    try delta_ptr.multiply(prev_activation_transposed.*, nabla_w_ptr);
}

const BackpropResult = struct {
    delta_nabla_weights: []linalg.Matrix,
    delta_nabla_biases: []linalg.Matrix,
};

fn free_matrices(allocator: std.mem.Allocator, buf: []linalg.Matrix) void {
    for (buf) |matrix| {
        linalg.free_matrix_data(allocator, matrix);
    }
    allocator.free(buf);
}

fn free_backprop_result(allocator: std.mem.Allocator, backprop_result: BackpropResult) void {
    free_matrices(allocator, backprop_result.delta_nabla_biases);
    free_matrices(allocator, backprop_result.delta_nabla_weights);
}

pub const DataPoint = struct {
    /// input (image pixels)
    x: []f64,
    /// expected output
    y: []f64,
};

pub const Network = struct {
    layer_sizes: []usize,

    // should be same length as biases
    weights: []linalg.Matrix,

    // column vectors
    biases: []linalg.Matrix,

    pub fn layer_count(self: Network) usize {
        return self.layer_sizes.len;
    }

    fn alloc_nabla_w(self: Network, allocator: std.mem.Allocator) ![]linalg.Matrix {
        const weights_copy = try allocator.alloc(linalg.Matrix, self.weights.len);

        for (self.weights) |weight_matrix, i| {
            const rows = weight_matrix.num_rows();
            const cols = weight_matrix.num_cols();
            try linalg.alloc_matrix_data(allocator, &weights_copy[i], rows, cols);
            weights_copy[i].zeroes();
        }

        return weights_copy;
    }

    fn alloc_nabla_b(self: Network, allocator: std.mem.Allocator) ![]linalg.Matrix {
        const biases_copy = try allocator.alloc(linalg.Matrix, self.biases.len);

        for (self.biases) |bias, i| {
            const rows = bias.num_rows();
            const cols = bias.num_cols();
            try linalg.alloc_matrix_data(allocator, &biases_copy[i], rows, cols);
            biases_copy[i].zeroes();
        }

        return biases_copy;
    }

    fn backprop(self: Network, allocator: std.mem.Allocator, point: DataPoint) !BackpropResult {
        var delta_nabla_w = try self.alloc_nabla_w(allocator);
        var delta_nabla_b = try self.alloc_nabla_b(allocator);

        var x_matrix = linalg.Matrix{
            .data = point.x,
            .rows = point.x.len,
            .cols = 1,
        };
        var y_matrix = linalg.Matrix{
            .data = point.y,
            .rows = point.y.len,
            .cols = 1,
        };

        var fed_forward = try self.feedforward(allocator, x_matrix);
        defer free_feedforward(allocator, fed_forward);

        var activations = fed_forward.activations;
        var z_results = fed_forward.z_results;

        // backward pass

        // this pointer is used to write results to our return value,
        // associated with nabla_b
        var delta_ptr: *linalg.Matrix = &delta_nabla_b[delta_nabla_b.len - 1];

        // cost_derivative(activations[-1], y)
        var activation_ptr = activations[fed_forward.activations.len - 1];
        try activation_ptr.sub(y_matrix, delta_ptr);

        // cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        var z_last = fed_forward.z_results[fed_forward.z_results.len - 1];
        maths.apply_sigmoid_prime_in_place(z_last.data);
        linalg.hadamard_product(delta_ptr.data, z_last.data, delta_ptr.data);

        // activations[-2].transpose()
        var prev_activation = &activations[activations.len - 2];
        var nabla_w_ptr = &delta_nabla_w[delta_nabla_w.len - 1];
        try delta_to_w(allocator, delta_ptr, prev_activation, nabla_w_ptr);

        var i: usize = 2;
        while (i < self.layer_sizes.len) {
            const z = z_results[z_results.len - i];
            var z_copy = try z.make_copy(allocator);
            defer linalg.free_matrix(allocator, z_copy);
            maths.apply_sigmoid_prime(z.data, z_copy.data);
            var w = self.weights[self.weights.len - i + 1];

            // w'
            var w_transposed = try linalg.alloc_matrix(allocator, w.cols, w.rows);
            defer linalg.free_matrix(allocator, w_transposed);
            linalg.transpose(w, w_transposed);

            // w' (dot) delta
            var new_delta = try linalg.alloc_matrix(allocator, w_transposed.num_rows(), delta_ptr.num_cols());
            defer linalg.free_matrix(allocator, new_delta);

            try w_transposed.multiply(delta_ptr.*, new_delta);

            // copy result back to delta_ptr
            delta_ptr = &delta_nabla_b[delta_nabla_b.len - i];
            new_delta.copy_data_unsafe(delta_ptr.data);

            nabla_w_ptr = &delta_nabla_w[delta_nabla_w.len - i];
            try delta_to_w(allocator, delta_ptr, &activations[activations.len - i - 1], nabla_w_ptr);

            i += 1;
        }

        return BackpropResult{ .delta_nabla_weights = delta_nabla_w, .delta_nabla_biases = delta_nabla_b };
    }

    /// Updates weights and biases with batch of data
    fn update_with_batch(self: Network, allocator: std.mem.Allocator, batch: []const DataPoint, eta: f64) !void {
        // TODO: use just one big matrix for each batch
        var nabla_w = try self.alloc_nabla_w(allocator);
        defer free_matrices(allocator, nabla_w);

        var nabla_b = try self.alloc_nabla_b(allocator);
        defer free_matrices(allocator, nabla_b);

        for (batch) |point| {
            const backprop_result = try self.backprop(allocator, point);
            defer free_backprop_result(allocator, backprop_result);

            // overwrite nabla_w, and nabla_b with deltas
            for (backprop_result.delta_nabla_weights) |delta_w, i| {
                try delta_w.add(nabla_w[i], &nabla_w[i]);
            }

            for (backprop_result.delta_nabla_biases) |delta_b, i| {
                try delta_b.add(nabla_b[i], &nabla_b[i]);
            }
        }

        // update weights, biases
        const scalar = eta / @intToFloat(f64, batch.len);
        for (self.weights) |_, i| {
            var weight = self.weights[i];
            nabla_w[i].scale(scalar);
            try weight.sub(nabla_w[i], &weight);
        }
        for (self.biases) |_, i| {
            var bias = self.biases[i];
            nabla_b[i].scale(scalar);
            try bias.sub(nabla_b[i], &bias);
        }
    }

    fn sgd_epoch(self: Network, allocator: std.mem.Allocator, train_data: []DataPoint, eta: f64) !void {
        const batch_size = 10;
        var i: usize = 0;
        while (i < train_data.len) {
            const remaining = train_data.len - 1 - i;
            const end_indx = if (remaining >= batch_size) i + batch_size else i + remaining;
            const batch_view = train_data[i..end_indx];
            try self.update_with_batch(allocator, batch_view, eta);
            i += batch_size;
        }
    }

    fn print_layer(self: Network, layer: usize) void {
        std.debug.print("weights at {}\n", .{layer});
        self.weights[layer].print();
        std.debug.print("biases at {}\n", .{layer});
        self.biases[layer].print();
    }

    pub fn sgd(self: Network, allocator: std.mem.Allocator, train_data: []DataPoint, eta: f64) !void {
        const EPOCHS = 10;
        var epoch: usize = 0;
        while (epoch < EPOCHS) {
            shuffle(DataPoint, train_data);
            std.debug.print("started epoch {} of {}\n", .{ epoch + 1, EPOCHS });
            try self.sgd_epoch(allocator, train_data, eta);
            std.debug.print("finished epoch {} of {}\n", .{ epoch + 1, EPOCHS });
            epoch += 1;
        }
    }

    /// Need to free result
    pub fn feedforward(self: Network, allocator: std.mem.Allocator, x_matrix: linalg.Matrix) !FeedforwardResult {
        // feedforward, and save the activations
        var activations = try allocator.alloc(linalg.Matrix, self.layer_count());
        for (activations) |_, i| {
            if (i == 0) {
                try linalg.alloc_matrix_data(allocator, &activations[i], x_matrix.num_rows(), x_matrix.num_cols());
            } else {
                const b = self.biases[i - 1];
                try linalg.alloc_matrix_data(allocator, &activations[i], b.num_rows(), b.num_cols());
            }
        }
        var z_results = try allocator.alloc(linalg.Matrix, self.layer_count() - 1);
        for (z_results) |_, i| {
            const b = self.biases[i];
            try linalg.alloc_matrix_data(allocator, &z_results[i], b.num_rows(), b.num_cols());
        }

        activations[0].copy_data_unsafe(x_matrix.data);
        var activation_ptr: linalg.Matrix = activations[0];
        for (self.weights) |w, i| {
            const b = self.biases[i];
            var next_activation = activations[i + 1];

            // dimension of activation = dimension of b

            // w * x
            try w.multiply(activation_ptr, &z_results[i]);
            // w * x + b = z
            try z_results[i].add(b, &z_results[i]);
            // sigmoid(w * x + b)
            maths.apply_sigmoid(z_results[i].data, next_activation.data);
            activation_ptr = next_activation;
        }

        // copy to result
        return FeedforwardResult{
            .activations = activations,
            .z_results = z_results,
        };
    }

    fn eval_point(self: Network, allocator: std.mem.Allocator, point: DataPoint) !bool {
        var x_matrix = linalg.Matrix{
            .data = point.x,
            .rows = point.x.len,
            .cols = 1,
        };
        var output = try self.feedforward(allocator, x_matrix);
        defer free_feedforward(allocator, output);
        var activations = output.activations;
        const digit = find_max_index(activations[activations.len - 1].data);
        const expected = find_max_index(point.y);
        return digit == expected;
    }

    pub fn evaluate(self: Network, allocator: std.mem.Allocator, test_data: []DataPoint) !usize {
        var num_correct: usize = 0;
        for (test_data) |point| {
            const is_correct = try self.eval_point(allocator, point);
            if (is_correct) {
                num_correct += 1;
            }
        }
        return num_correct;
    }
};

fn find_max_index(buf: []f64) usize {
    var max_i: usize = 0;
    var max: f64 = buf[0];
    for (buf) |val, i| {
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

/// Assumed to be the same length
fn copy_image_data(input: []const u8, output: []f64) void {
    for (input) |pixel, i| {
        output[i] = @intToFloat(f64, pixel);
    }
}

pub fn make_mnist_data_points(allocator: std.mem.Allocator, x: []const u8, x_chunk_size: usize, y: []const u8, y_output_size: usize) ![]DataPoint {
    const result = try allocator.alloc(DataPoint, x.len / x_chunk_size);
    var i: usize = 0;
    var idx: usize = 0;
    // assuming x / x_chunk_size and y.len are the same
    while (i < x.len) {
        //std.debug.print("i: {}\n", .{i});
        const slice = x[i .. i + x_chunk_size];
        const x_buffer = try allocator.alloc(f64, x_chunk_size);
        copy_image_data(slice, x_buffer);
        const y_buffer = try allocator.alloc(f64, y_output_size);
        write_digit(y[idx], y_buffer);

        result[idx] = DataPoint{
            .x = x_buffer,
            .y = y_buffer,
        };
        idx += 1;
        i += x_chunk_size;
    }
    return result;
}

pub fn free_mnist_data_points(allocator: std.mem.Allocator, data: []DataPoint) void {
    for (data) |data_pt| {
        allocator.free(data_pt.x);
        allocator.free(data_pt.y);
    }
    allocator.free(data);
}

fn sum_sizes(sizes: []const usize) usize {
    var result: usize = 0;
    for (sizes) |size| {
        result += size;
    }
    return result;
}

pub fn alloc_network(allocator: std.mem.Allocator, layer_sizes: []const usize) error{OutOfMemory}!*Network {
    var network = try allocator.create(Network);
    network.layer_sizes = try allocator.alloc(usize, layer_sizes.len);
    var biases_weights_len = layer_sizes.len - 1;
    network.biases = try allocator.alloc(linalg.Matrix, biases_weights_len);
    network.weights = try allocator.alloc(linalg.Matrix, biases_weights_len);

    // allocate weights and biases matrices
    for (layer_sizes) |layer_size, i| {
        // copy layer sizes
        network.layer_sizes[i] = layer_size;
        if (i == 0) {
            continue;
        }
        const prev_layer_size: usize = layer_sizes[i - 1];
        try linalg.alloc_matrix_data(allocator, &network.weights[i - 1], layer_size, prev_layer_size);
        try linalg.alloc_matrix_data(allocator, &network.biases[i - 1], layer_size, 1);
    }

    return network;
}

pub fn free_network(allocator: std.mem.Allocator, network: *Network) void {
    for (network.weights) |weight_matrix| {
        linalg.free_matrix_data(allocator, weight_matrix);
    }
    for (network.biases) |bias_matrix| {
        linalg.free_matrix_data(allocator, bias_matrix);
    }
    allocator.free(network.weights);
    allocator.free(network.biases);
    allocator.free(network.layer_sizes);
    allocator.destroy(network);
}

test "feedforward test" {
    const allocator = std.testing.allocator;
    var layer_sizes = [_]usize{ 2, 2, 2 };
    var network = try alloc_network(allocator, &layer_sizes);
    defer free_network(allocator, network);
    var w_1 = [_]f64{
        1, 0,
        0, 1,
    };
    var b_1 = [_]f64{
        0.5,
        0.5,
    };
    network.weights[0].copy_data_unsafe(&w_1);
    network.biases[0].copy_data_unsafe(&b_1);

    var w_2 = [_]f64{
        -1, 0,
        0,  1,
    };
    var b_2 = [_]f64{
        0.2,
        0.2,
    };

    network.weights[1].copy_data_unsafe(&w_2);
    network.biases[1].copy_data_unsafe(&b_2);
    var input_x = [_]f64{
        0.1,
        0.1,
    };
    var input = linalg.Matrix{
        .data = &input_x,
        .rows = 2,
        .cols = 1,
    };
    const TOLERANCE = 1e-9;

    var res = try network.feedforward(allocator, input);
    defer free_feedforward(allocator, res);

    // var output = network.output_layer();
    var expected_out = [_]f64{ 0.3903940131009935, 0.6996551604890665 };
    var activation = res.activations[res.activations.len - 1];
    try std.testing.expectApproxEqRel(expected_out[0], activation.data[0], TOLERANCE);
    try std.testing.expectApproxEqRel(expected_out[1], activation.data[1], TOLERANCE);
}

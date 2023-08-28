const std = @import("std");
const linalg = @import("linalg.zig");
const maths = @import("maths.zig");
const perf = @import("performance.zig");

/// https://stackoverflow.com/a/73020142
fn range(len: usize) []const void {
    return @as([*]void, undefined)[0..len];
}

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
var rand = randgen.init(0);

fn shuffle(comptime T: type, arr: []T) void {
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
    /// @internal weights transposed
    weights_t: []linalg.Matrix,

    // column vectors
    biases: []linalg.Matrix,

    /// @internal
    activations: []linalg.Matrix,
    /// @internal activations transpoed
    activations_t: []linalg.Matrix,

    /// @internal
    z_results: []linalg.Matrix,

    /// @internal mutated
    delta_nabla_w: []linalg.Matrix,
    delta_nabla_b: []linalg.Matrix,

    pub fn layer_count(self: Network) usize {
        return self.layer_sizes.len;
    }

    /// Sets all weights/biases to 0. Used for testing.
    pub fn init_zeros(self: Network) void {
        for (self.biases) |bias| {
            bias.set_all(0);
        }
        for (self.weights) |weight| {
            weight.set_all(0);
        }
    }

    /// Initialize random weights with standard normal
    /// distribution
    pub fn init_randn(self: Network) void {
        for (self.biases) |bias| {
            for (range(bias.data.len)) |_, i| {
                bias.data[i] = rand.random().floatNorm(f64);
            }
        }
        for (self.weights) |weight| {
            for (range(weight.data.len)) |_, i| {
                weight.data[i] = rand.random().floatNorm(f64);
            }
        }
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
        var delta_nabla_w = self.delta_nabla_w;
        var delta_nabla_b = self.delta_nabla_b;

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

        try self.feedforward_mut(x_matrix);

        var activations = self.activations;
        var z_results = self.z_results;

        // backward pass

        // this pointer is used to write results to our return value,
        // associated with nabla_b
        var delta_ptr: *linalg.Matrix = &delta_nabla_b[delta_nabla_b.len - 1];

        // cost_derivative(activations[-1], y)
        var activation_ptr = activations[self.activations.len - 1];
        try activation_ptr.sub(y_matrix, delta_ptr);

        // cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        var z_last = self.z_results[self.z_results.len - 1];
        maths.apply_sigmoid_prime_in_place(z_last.data);
        linalg.hadamard_product(delta_ptr.data, z_last.data, delta_ptr.data);

        // activations[-2].transpose()
        var prev_activation = activations[activations.len - 2];
        var nabla_w_ptr = &delta_nabla_w[delta_nabla_w.len - 1];
        var prev_activation_t = self.activations_t[activations.len - 2];
        linalg.transpose(prev_activation, &prev_activation_t);
        delta_ptr.multiply_unsafe(prev_activation_t, nabla_w_ptr);

        var i: usize = 2;
        while (i < self.layer_sizes.len) {
            const z = z_results[z_results.len - i];
            var z_copy = try z.make_copy(allocator);
            defer linalg.free_matrix(allocator, z_copy);
            maths.apply_sigmoid_prime(z.data, z_copy.data);
            var w = self.weights[self.weights.len - i + 1];

            // w'
            // var w_transposed = try linalg.alloc_matrix(allocator, w.cols, w.rows);
            var w_transposed = self.weights_t[self.weights_t.len - i + 1];
            // defer linalg.free_matrix(allocator, w_transposed);
            linalg.transpose(w, &w_transposed);

            // w' (dot) delta
            var new_delta = try linalg.alloc_matrix(allocator, w_transposed.num_rows(), delta_ptr.num_cols());
            defer linalg.free_matrix(allocator, new_delta);

            try w_transposed.multiply(delta_ptr.*, new_delta);

            // copy result back to delta_ptr
            delta_ptr = &delta_nabla_b[delta_nabla_b.len - i];
            new_delta.copy_data_unsafe(delta_ptr.data);

            nabla_w_ptr = &delta_nabla_w[delta_nabla_w.len - i];
            prev_activation_t = self.activations_t[activations.len - i - 1];
            linalg.transpose(activations[activations.len - i - 1], &prev_activation_t);
            delta_ptr.multiply_unsafe(prev_activation_t, nabla_w_ptr);

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

            // overwrite nabla_w, and nabla_b with deltas
            for (backprop_result.delta_nabla_weights) |delta_w, i| {
                try nabla_w[i].add(delta_w, &nabla_w[i]);
            }

            for (backprop_result.delta_nabla_biases) |delta_b, i| {
                try nabla_b[i].add(delta_b, &nabla_b[i]);
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

    pub fn print_biases(self: Network, layer: usize) void {
        std.debug.print("biases at {}\n", .{layer});
        self.biases[layer].print();
    }

    fn print_weights(self: Network, layer: usize) void {
        std.debug.print("weights at {}\n", .{layer});
        self.weights[layer].print();
    }

    fn print_layer(self: Network, layer: usize) void {
        self.print_weights(layer);
        self.print_biases(layer);
    }

    pub fn sgd(self: Network, allocator: std.mem.Allocator, train_data: []DataPoint, eta: f64, epochs: usize) !void {
        var epoch: usize = 0;
        var stopwatch = perf.Stopwatch{
            .last_ts = 0,
        };
        while (epoch < epochs) {
            shuffle(DataPoint, train_data);
            stopwatch.start();
            std.debug.print("started epoch {} of {}\n", .{ epoch + 1, epochs });
            try self.sgd_epoch(allocator, train_data, eta);
            stopwatch.report("epoch finished");
            std.debug.print("finished epoch {} of {}\n", .{ epoch + 1, epochs });
            epoch += 1;
        }
    }

    fn feedforward_mut(self: Network, x_matrix: linalg.Matrix) !void {
        self.activations[0].copy_data_unsafe(x_matrix.data);
        var activation_ptr: linalg.Matrix = self.activations[0];
        for (self.weights) |w, i| {
            const b = self.biases[i];
            var next_activation = self.activations[i + 1];

            // dimension of activation = dimension of b

            // w * x
            try w.multiply(activation_ptr, &self.z_results[i]);
            // w * x + b = z
            try self.z_results[i].add(b, &self.z_results[i]);
            // sigmoid(w * x + b)
            maths.apply_sigmoid(self.z_results[i].data, next_activation.data);
            activation_ptr = next_activation;
        }
    }

    /// Used for testing.
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
    network.weights_t = try allocator.alloc(linalg.Matrix, biases_weights_len);

    // allocate weights and biases matrices
    for (layer_sizes) |layer_size, i| {
        // copy layer sizes
        network.layer_sizes[i] = layer_size;
        if (i == 0) {
            continue;
        }
        const prev_layer_size: usize = layer_sizes[i - 1];
        try linalg.alloc_matrix_data(allocator, &network.weights[i - 1], layer_size, prev_layer_size);
        try linalg.alloc_matrix_data(allocator, &network.weights_t[i - 1], prev_layer_size, layer_size);
        try linalg.alloc_matrix_data(allocator, &network.biases[i - 1], layer_size, 1);
    }

    // allocate activations
    var activations = try allocator.alloc(linalg.Matrix, network.layer_count());
    var activations_t = try allocator.alloc(linalg.Matrix, network.layer_count());
    for (activations) |_, i| {
        try linalg.alloc_matrix_data(allocator, &activations[i], layer_sizes[i], 1);
        try linalg.alloc_matrix_data(allocator, &activations_t[i], 1, layer_sizes[i]);
    }
    network.activations = activations;
    network.activations_t = activations_t;
    var z_results = try allocator.alloc(linalg.Matrix, network.layer_count() - 1);
    for (z_results) |_, i| {
        const b = network.biases[i];
        try linalg.alloc_matrix_data(allocator, &z_results[i], b.num_rows(), b.num_cols());
    }
    network.z_results = z_results;

    // allocate intermediate backprop results
    var delta_nabla_w = try network.alloc_nabla_w(allocator);
    var delta_nabla_b = try network.alloc_nabla_b(allocator);
    network.delta_nabla_b = delta_nabla_b;
    network.delta_nabla_w = delta_nabla_w;

    return network;
}

pub fn free_network(allocator: std.mem.Allocator, network: *Network) void {
    free_matrices(allocator, network.delta_nabla_b);
    free_matrices(allocator, network.delta_nabla_w);
    for (network.activations) |activation| {
        linalg.free_matrix_data(allocator, activation);
    }
    allocator.free(network.activations);
    for (network.activations_t) |activation_t| {
        linalg.free_matrix_data(allocator, activation_t);
    }
    allocator.free(network.activations_t);
    for (network.z_results) |z| {
        linalg.free_matrix_data(allocator, z);
    }
    allocator.free(network.z_results);
    for (network.weights) |weight_matrix| {
        linalg.free_matrix_data(allocator, weight_matrix);
    }
    allocator.free(network.weights);
    for (network.weights_t) |w_t| {
        linalg.free_matrix_data(allocator, w_t);
    }
    allocator.free(network.weights_t);
    for (network.biases) |bias_matrix| {
        linalg.free_matrix_data(allocator, bias_matrix);
    }
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

test "backpropagation test" {
    const allocator = std.testing.allocator;
    var x = [_]f64{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 18, 18, 18, 126, 136, 175, 26, 166, 255, 247, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 36, 94, 154, 170, 253, 253, 253, 253, 253, 225, 172, 253, 242, 195, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 49, 238, 253, 253, 253, 253, 253, 253, 253, 253, 251, 93, 82, 82, 56, 39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 219, 253, 253, 253, 253, 253, 198, 182, 247, 241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 156, 107, 253, 253, 205, 11, 0, 43, 154, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 1, 154, 253, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 139, 253, 190, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 190, 253, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 241, 225, 160, 108, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 240, 253, 253, 119, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 186, 253, 253, 150, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 93, 252, 253, 187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 249, 253, 249, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 130, 183, 253, 253, 207, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 148, 229, 253, 253, 253, 250, 182, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 114, 221, 253, 253, 253, 253, 201, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 66, 213, 253, 253, 253, 253, 198, 81, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 171, 219, 253, 253, 253, 253, 195, 80, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 172, 226, 253, 253, 253, 253, 244, 133, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 136, 253, 253, 253, 212, 135, 132, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    var y = [_]f64{ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 };
    const data_point = DataPoint{
        .x = &x,
        .y = &y,
    };
    const image_size = 28 * 28;
    const HIDDEN_LAYER_SIZE = 30;
    const DIGITS = 10;
    const layer_sizes = [_]usize{ image_size, HIDDEN_LAYER_SIZE, DIGITS };
    var network = try alloc_network(allocator, &layer_sizes);
    defer free_network(allocator, network);
    network.init_zeros();
    const result = try network.backprop(allocator, data_point);

    var expected_delta_b = [_]f64{ 0.125, 0.125, 0.125, 0.125, 0.125, -0.125, 0.125, 0.125, 0.125, 0.125 };
    try std.testing.expectEqualSlices(f64, &expected_delta_b, result.delta_nabla_biases[1].data);

    // expected weights for reference
    //     [[ 0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625]
    //  [ 0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625]
    //  [ 0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625]
    //  [ 0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625]
    //  [ 0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625]
    //  [-0.0625 -0.0625 -0.0625 -0.0625 -0.0625 -0.0625 -0.0625 -0.0625 -0.0625
    //   -0.0625 -0.0625 -0.0625 -0.0625 -0.0625 -0.0625 -0.0625 -0.0625 -0.0625
    //   -0.0625 -0.0625 -0.0625 -0.0625 -0.0625 -0.0625 -0.0625 -0.0625 -0.0625
    //   -0.0625 -0.0625 -0.0625]
    //  [ 0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625]
    //  [ 0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625]
    //  [ 0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625]
    //  [ 0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625]]
}

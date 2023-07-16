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

const BackpropResult = struct {
    delta_nabla_weights: []linalg.Matrix,
    delta_nabla_biases: []linalg.Matrix,
};

pub const DataPoint = struct {
    /// input (image pixels)
    x: []const f32,
    /// expected output
    y: []const f32,
};

pub const Network = struct {
    allocator: std.mem.Allocator,

    layer_sizes: []usize,

    // should be same length as biases
    weights: []linalg.Matrix,

    // column vectors
    biases: []linalg.Matrix,

    pub fn layer_count(self: Network) usize {
        return self.layer_sizes.len;
    }

    fn alloc_nabla_w(self: Network) ![]linalg.Matrix {
        const weights_copy = try self.allocator.alloc(linalg.Matrix, self.weights.len);

        for (self.weights) |weight_matrix, i| {
            const rows = weight_matrix.num_rows();
            const cols = weight_matrix.num_cols();
            try linalg.alloc_matrix_data(self.allocator, weights_copy[i], rows, cols);
            weights_copy[i].zeroes();
        }

        return weights_copy;
    }

    fn alloc_nabla_b(self: Network) ![]linalg.Matrix {
        const biases_copy = try self.allocator.alloc(linalg.Matrix, self.biases.len);

        for (self.biases) |bias, i| {
            const rows = bias.num_rows();
            const cols = bias.num_cols();
            try linalg.alloc_matrix_data(self.allocator, biases_copy[i], rows, cols);
            biases_copy[i].zeroes();
        }

        return biases_copy;
    }

    fn free_nabla(self: Network, buf: []linalg.Matrix) void {
        for (buf) |matrix| {
            linalg.free_matrix_data(self.allocator, matrix);
        }
        self.allocator.free(buf);
    }

    fn backprop(self: Network, point: DataPoint) !BackpropResult {
        var delta_nabla_w = try self.alloc_nabla_w();
        var delta_nabla_b = try self.alloc_nabla_b();

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

        // feedforward, and save the activations
        var activations = try self.allocator.alloc(linalg.Matrix, self.layer_count());
        var z_results = try self.allocator.alloc(linalg.Matrix, self.layer_count() - 1);
        var activation_ptr: *linalg.Matrix = &x_matrix;
        activations[0] = x_matrix;
        for (self.weights) |w, i| {
            const b = self.biases[i];
            var next_activation = activations[i + 1];

            // dimension of activation = dimension of b
            linalg.alloc_matrix_data(self.allocator, next_activation, b.num_rows(), b.num_cols());
            linalg.alloc_matrix_data(self.allocator, z_results[i], b.num_rows(), b.num_cols());

            // w * x
            try w.multiply(activation_ptr.*, &z_results[i]);
            // w * x + b = z
            try z_results[i].add(b, &z_results[i]);
            // sigmoid(w * x + b)
            maths.apply_sigmoid(z_results, next_activation.data);
            activation_ptr = &next_activation;
        }

        // backward pass

        // cost derivative
        var delta_ptr: *linalg.Matrix = &delta_nabla_b[delta_nabla_b.len - 1];
        try activation_ptr.sub(y_matrix, delta_ptr);
        // TODO: make these all matrix operations
        maths.apply_sigmoid_prime_in_place(z_results[z_results.len - 1].data);
        linalg.hadamard_product(delta_ptr.data, z_results[z_results.len - 1].data, delta_ptr.data);

        // garbage collection
        for (activations) |activation| {
            linalg.free_matrix_data(self.allocator, activation);
        }
        defer self.allocator.free(activations);

        for (z_results) |z_vec| {
            linalg.free_matrix_data(self.allocator, z_vec);
        }
        defer self.allocator.free(z_results);

        return BackpropResult{ .delta_nabla_weights = delta_nabla_w, .delta_nabla_biases = delta_nabla_b };
    }

    /// Updates weights and biases with batch of data
    fn update_with_batch(self: Network, batch: []const DataPoint, eta: f32) !void {
        // TODO: use just one big matrix for each batch
        var nabla_w = try self.alloc_nabla_w();
        defer self.free_nabla(nabla_w);

        var nabla_b = try self.alloc_nabla_b();
        defer self.free_nabla(nabla_b);

        for (batch) |point| {
            const backprop_result = try self.backprop(point);
            // overwrite nabla_w, and nabla_b with deltas
            for (backprop_result.delta_nabla_weights) |delta_w, i| {
                delta_w.add(nabla_w[i], nabla_w[i]);
            }

            for (backprop_result.delta_nabla_biases) |delta_b, i| {
                delta_b.add(nabla_b[i], nabla_b[i]);
            }
        }

        // update weights, biases
        const scalar = eta / @intToFloat(f32, batch.len);
        for (self.weights) |weight, i| {
            nabla_w[i].scale(scalar);
            linalg.subtract(weight, nabla_w[i], weight);
        }
        for (self.biases) |bias, i| {
            nabla_b[i].scale(scalar);
            linalg.subtract(bias, nabla_b[i], bias);
        }
    }
};

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
    network.layer_biases = try allocator.alloc(f32, sum_sizes(layer_sizes[1..]));
    network.layer_weights = try allocator.alloc(linalg.Matrix, layer_sizes.len - 1);

    // allocate weight matrices
    for (layer_sizes) |layer_size, i| {
        // copy layer sizes
        network.layer_sizes[i] = layer_size;
        if (i == 0) {
            continue;
        }
        const prev_layer_size: usize = layer_sizes[i - 1];
        try linalg.alloc_matrix_data(allocator, &network.layer_weights[i - 1], layer_size, prev_layer_size);
    }

    return network;
}

pub fn free_network(allocator: std.mem.Allocator, network: *Network) void {
    for (network.layer_weights) |weight_matrix| {
        linalg.free_matrix_data(allocator, &weight_matrix);
    }
    allocator.free(network.layer_weights);
    allocator.free(network.layer_biases);
    allocator.free(network.layer_sizes);
    allocator.destroy(network);
}

test "biases vector access test" {
    var layer_biases = [_]f32{
        1, 2, 3, 4, 5, 6,
    };
    var layer_sizes = [_]usize{
        4, 3, 2, 1,
    };

    const allocator = std.testing.allocator;
    var network = try alloc_network(allocator, &layer_sizes);
    defer free_network(allocator, network);
    network.set_biases(&layer_biases);
    const first_biases = try network.biases_at_layer(1);
    var expected = [_]f32{
        1, 2, 3,
    };
    try std.testing.expectEqualSlices(f32, first_biases, &expected);
}

// test "feedforward test" {
//     var w_1 = [_]f32{
//         1, 0,
//         0, 1,
//     };
//     var b_1 = [_]f32{
//         0.5,
//         0.5,
//     };
//     var activations_1 = [_]f32{ 0, 0 };
//     var z_vector_1 = [_]f32{ 0, 0 };
//     var layer_1 = NetworkLayer{
//         .weights = linalg.Matrix{
//             .data = &w_1,
//             .rows = 2,
//             .cols = 2,
//         },
//         .biases = &b_1,
//         .activations = &activations_1,
//         .z_vector = &z_vector_1,
//     };
//
//     var w_2 = [_]f32{
//         -1, 0,
//         0,  1,
//     };
//     var b_2 = [_]f32{
//         0.2,
//         0.2,
//     };
//     var activations_2 = [_]f32{ 0, 0 };
//     var z_vector_2 = [_]f32{ 0, 0 };
//     var layer_2 = NetworkLayer{
//         .weights = linalg.Matrix{
//             .data = &w_2,
//             .rows = 2,
//             .cols = 2,
//         },
//         .biases = &b_2,
//         .activations = &activations_2,
//         .z_vector = &z_vector_2,
//     };
//     var layers = [_]NetworkLayer{
//         layer_1,
//         layer_2,
//     };
//     var input_layer = [_]f32{
//         0.1,
//         0.1,
//     };
//     var network = Network{
//         .layers = &layers,
//     };
//     network.feedforward(&input_layer);
//
//     var output = network.output_layer();
//     var expected_out = [_]f32{ 0.3903940131009935, 0.6996551604890665 };
//     try std.testing.expectApproxEqRel(expected_out[0], output[0], 1e-6);
//     try std.testing.expectApproxEqRel(expected_out[1], output[1], 1e-6);
// }

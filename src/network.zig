const std = @import("std");
const linalg = @import("linalg.zig");
const maths = @import("maths.zig");

pub const NetworkLayer = struct {
    weights: linalg.Matrix,
    biases: []f32,
    activations: []f32,
    z_vector: []f32,

    /// Returns size of the weight matrix
    /// (number of elements)
    pub fn size(self: NetworkLayer) usize {
        return self.weights.size();
    }

    pub fn forward_pass(self: NetworkLayer, x: []const f32) void {
        // w*x
        self.weights.apply(x, self.z_vector);
        // w*x + b
        linalg.accumulate(self.z_vector, self.biases);
        // sigma(w*x + b)
        maths.apply_sigmoid(self.z_vector, self.activations);
    }
};

pub const GradientResult = struct {
    biases: []f32,
    weights: linalg.Matrix,
};

pub const Network = struct {
    layer_sizes: []usize,

    layer_weights: []linalg.Matrix,

    /// all in one array, but accessed separately
    layer_biases: []f32,

    pub fn layer_count(self: Network) usize {
        return self.layer_sizes.len;
    }

    pub fn set_biases(self: Network, biases_vector: []const f32) void {
        for (biases_vector) |b, i| {
            self.layer_biases[i] = b;
        }
    }

    /// Returns a slice of biases at the given layer.
    /// Note that the input layer (at 0) has no biases.
    pub fn biases_at_layer(self: Network, i: usize) error{IndexError}![]f32 {
        if (i == 0 or i >= self.layer_count()) {
            return error.IndexError;
        }
        const adjusted_i = i - 1;
        var to_skip_over: usize = 0;
        var j: usize = 1;
        while (j < adjusted_i) : (j += 1) {
            const size = self.layer_sizes[j];
            to_skip_over += size;
        }
        const desired_size = self.layer_sizes[j];
        std.debug.print("skip: {}, size: {}\n", .{ to_skip_over, desired_size });
        return self.layer_biases[to_skip_over..desired_size];
    }

    // /// Returns slice to the activations at a given layer
    // pub fn activations_at(self: Network, i: usize) []f32 {
    //     return self.layers[i].activations;
    // }

    // pub fn z_vector_at(self: Network, i: usize) []f32 {
    //     return self.layers[i].z_vector;
    // }

    // pub fn output_layer(self: Network) []f32 {
    //     return self.activations_at(self.layer_count() - 1);
    // }

    // pub fn feedforward(self: Network, input_layer: []const f32) void {
    //     for (self.layers) |layer, i| {
    //         if (i == 0) {
    //             layer.forward_pass(input_layer);
    //         } else {
    //             var prev_layer_activations = self.activations_at(i - 1);
    //             layer.forward_pass(prev_layer_activations);
    //         }
    //     }
    // }

    // pub fn backprop(self: Network, input_layer: []const f32, y: []const f32, out: []GradientResult) error{ LayerDimensionError, MatrixDimensionError, OutOfMemory }!void {
    //     self.feedforward(input_layer);
    //     var output = self.output_layer();
    //     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    //     const allocator = gpa.allocator();
    //     const layer_size = y.len;

    //     // get cost derivative (a - y)
    //     var cost_derivative = try allocator.alloc(f32, layer_size);
    //     defer allocator.free(cost_derivative);
    //     linalg.subtract(output, y, cost_derivative);

    //     // get sigmoid_prime
    //     var sigmoid_primes = try allocator.alloc(f32, layer_size);
    //     defer allocator.free(sigmoid_primes);
    //     maths.apply_sigmoid_prime(self.z_vector_at(self.layer_count() - 1), sigmoid_primes);

    //     var out_ptr = out[out.len - 1];

    //     // save product to output biases (already allocated)
    //     linalg.hadamard_product(cost_derivative, sigmoid_primes, out_ptr.biases);
    //     var delta = out_ptr.biases;

    //     // transpose activations[-2]
    //     var prev_activations = self.activations_at(self.layer_count() - 2);

    //     if (delta.len != prev_activations.len) {
    //         std.debug.print("Error got different lengths, delta.len {}, prev_activations.len {}\n", .{ delta.len, prev_activations.len });
    //         return error.LayerDimensionError;
    //     }

    //     var activations_transposed = try allocator.alloc(f32, prev_activations.len);
    //     defer allocator.free(activations_transposed);
    //     var activations_matrix = linalg.Matrix{
    //         .data = prev_activations,
    //         .rows = prev_activations.len,
    //         .cols = 1,
    //     };
    //     var activations_transposed_mat = linalg.Matrix{
    //         .data = activations_transposed,
    //         .rows = 1,
    //         .cols = prev_activations.len,
    //     };

    //     linalg.transpose(activations_matrix, &activations_transposed_mat);
    //     // store delta into a matrix
    //     var delta_col_vec = linalg.Matrix{
    //         .data = delta,
    //         .rows = delta.len,
    //         .cols = 1,
    //     };

    //     // delta (dot) activations[-2] becomes delta.len x activations.len
    //     // dimensional matrix, which are the weights
    //     var out_weight_matrix = out_ptr.weights;
    //     // std.debug.print("out length: {}, delta_rows: {}, activations len: {}\n", .{ out_weight_matrix.data.len, delta_col_vec.rows, activations_transposed_mat.data.len });
    //     try delta_col_vec.multiply(activations_transposed_mat, &out_weight_matrix);

    //     // self.layer_count() excludes input layer, so + 1 to adjust
    //     const layer_count_adjusted = self.layer_count() - 1;
    //     var l: usize = layer_count_adjusted;
    //     while (l >= 0) : (l -= 1) {
    //         out_ptr = out[l];
    //         out_weight_matrix = out_ptr.weights;

    //         // sigmoid(z)
    //         var z_vector = self.z_vector_at(l);
    //         var sigmoid_primes_buf = try allocator.alloc(f32, z_vector.len);
    //         defer allocator.free(sigmoid_primes_buf);
    //         maths.apply_sigmoid_prime(z_vector, sigmoid_primes_buf);

    //         // delta = weights[l+1].transpose() (dot) delta[l+1] (*) sigmoid_prime(z)
    //         var weight_ahead = out[l + 1].weights;
    //         var weight_ahead_t_data = try allocator.alloc(f32, weight_ahead.num_rows() * weight_ahead.num_cols());
    //         var weight_ahead_t = linalg.Matrix{
    //             .data = weight_ahead_t_data,
    //             .rows = 0,
    //             .cols = 0,
    //         };
    //         linalg.transpose(weight_ahead, &weight_ahead_t);

    //         var delta_buf = try allocator.alloc(f32, z_vector.len);
    //         defer allocator.free(delta_buf);

    //         var delta_ahead = out[l + 1].biases;
    //         weight_ahead_t.apply(delta_ahead, delta_buf);

    //         linalg.hadamard_product(delta_buf, sigmoid_primes_buf, out_ptr.biases);
    //         var activations_behind_data = self.activations_at(l - 1);
    //         var activations_behind_t = linalg.Matrix{
    //             .data = activations_behind_data,
    //             .rows = 1,
    //             .cols = activations_behind_data.len,
    //         };

    //         var delta_current = out_ptr.biases;
    //         var delta_vec = linalg.Matrix{
    //             .data = delta_current,
    //             .rows = delta_current.len,
    //             .cols = 1,
    //         };
    //         try delta_vec.multiply(activations_behind_t, &out_ptr.weights);
    //     }
    // }
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

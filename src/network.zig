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
    weights: []linalg.Matrix,
};

pub const Network = struct {
    /// Includes output layer, excludes input layer
    layers: []NetworkLayer,

    /// Excludes input layer
    pub fn layer_count(self: Network) usize {
        return self.layers.len;
    }

    /// Returns slice to the activations at a given layer
    pub fn activations_at(self: Network, i: usize) []f32 {
        return self.layers[i].activations;
    }

    pub fn z_vector_at(self: Network, i: usize) []f32 {
        return self.layers[i].z_vector;
    }

    pub fn output_layer(self: Network) []f32 {
        return self.activations_at(self.layer_count() - 1);
    }

    pub fn feedforward(self: Network, input_layer: []f32) void {
        for (self.layers) |layer, i| {
            if (i == 0) {
                layer.forward_pass(input_layer);
            } else {
                var prev_layer_activations = self.activations_at(i - 1);
                layer.forward_pass(prev_layer_activations);
            }
        }
    }

    pub fn backprop(self: Network, input_layer: []const f32, y: []const f32, out: []GradientResult) void {
        self.feedforward(input_layer);
        var output = self.output_layer();
        var allocator = std.heap.page_allocator;
        var layer_size = y.len;

        // get cost derivative (a - y)
        var cost_derivative = try allocator.alloc(f32, layer_size);
        defer allocator.free(cost_derivative);
        linalg.subtract(output, y, cost_derivative);

        // get sigmoid_prime
        var sigmoid_primes = try allocator.alloc(f32, layer_size);
        defer allocator.free(sigmoid_primes);
        maths.apply_sigmoid_prime(self.z_vector_at(self.layer_count() - 1), sigmoid_primes);

        var delta = out[out.len - 1].biases;
        linalg.hadamard_product(cost_derivative, sigmoid_primes, delta);

        // TODO: compute and assign to out.weights

        var l: usize = self.layer_count() - 2;
        while (l > 0) : (l -= 1) {
            var z_vector = self.z_vector_at(l);
            // reuse buffer
            maths.apply_sigmoid_prime(z_vector, sigmoid_primes);
            delta = out[l].biases;
            // TODO: check that this doesn't overwrite results
        }
    }
};

test "feedforward test" {
    var w_1 = [_]f32{
        1, 0,
        0, 1,
    };
    var b_1 = [_]f32{
        0.5,
        0.5,
    };
    var activations_1 = [_]f32{ 0, 0 };
    var z_vector_1 = [_]f32{ 0, 0 };
    var layer_1 = NetworkLayer{
        .weights = linalg.Matrix{
            .data = &w_1,
            .rows = 2,
            .cols = 2,
        },
        .biases = &b_1,
        .activations = &activations_1,
        .z_vector = &z_vector_1,
    };

    var w_2 = [_]f32{
        -1, 0,
        0,  1,
    };
    var b_2 = [_]f32{
        0.2,
        0.2,
    };
    var activations_2 = [_]f32{ 0, 0 };
    var z_vector_2 = [_]f32{ 0, 0 };
    var layer_2 = NetworkLayer{
        .weights = linalg.Matrix{
            .data = &w_2,
            .rows = 2,
            .cols = 2,
        },
        .biases = &b_2,
        .activations = &activations_2,
        .z_vector = &z_vector_2,
    };
    var layers = [_]NetworkLayer{
        layer_1,
        layer_2,
    };
    var input_layer = [_]f32{
        0.1,
        0.1,
    };
    var network = Network{
        .layers = &layers,
    };
    network.feedforward(&input_layer);

    var output = network.output_layer();
    var expected_out = [_]f32{ 0.3903940131009935, 0.6996551604890665 };
    try std.testing.expectApproxEqRel(expected_out[0], output[0], 1e-6);
    try std.testing.expectApproxEqRel(expected_out[1], output[1], 1e-6);
}

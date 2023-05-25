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

    pub fn feedforward(self: NetworkLayer, x: []const f32) void {
        // w*x
        self.weights.apply(x, self.z_vector);
        // w*x + b
        linalg.accumulate(self.z_vector, self.biases);
        // sigma(w*x + b)
        maths.apply_sigmoid(self.z_vector, self.activations);
    }
};

pub const Network = struct {
    layers: []NetworkLayer,
    input_layer: []f32,

    pub fn layer_count(self: Network) usize {
        return self.layers.len;
    }

    /// Returns slice to the activations at a given layer
    pub fn activations_at(self: Network, i: usize) []f32 {
        return self.layers[i].activations;
    }

    pub fn feedforward(self: Network) void {
        for (self.layers) |layer, i| {
            if (i == 0) {
                layer.feedforward(self.input_layer);
            } else {
                var prev_layer_activations = self.activations_at(i - 1);
                layer.feedforward(prev_layer_activations);
            }
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
        .input_layer = &input_layer,
    };
    network.feedforward();

    var output = network.activations_at(network.layer_count() - 1);
    var expected_out = [_]f32{ 0.3903940131009935, 0.6996551604890665 };
    try std.testing.expectApproxEqRel(expected_out[0], output[0], 1e-6);
    try std.testing.expectApproxEqRel(expected_out[1], output[1], 1e-6);
}

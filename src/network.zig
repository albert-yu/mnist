const std = @import("std");

fn sigmoid(val: f32) f32 {
    return 1 / (1 + std.math.exp(-val));
}

pub const NetworkLayer = struct {
    weights: []f32,
    biases: []f32,

    /// Returns size of the weight matrix
    /// (number of elements)
    pub fn size(self: NetworkLayer) usize {
        return self.weights.len;
    }

    pub fn init(input_count: usize, output_count: usize) !NetworkLayer {
        var weight_size = input_count * output_count;
        var weights = [_]f32{0} ** weight_size;
        var biases = [_]f32{0} ** output_count;
        return NetworkLayer{
            .weights = weights,
            .biases = biases,
        };
    }

    /// w * x + b
    fn multiply_add(self: NetworkLayer, x: []const f32) f32 {
        var result: f32 = 0;
        for (self.weights) |weight, i| {
            const other_val = x[i];
            result += weight * other_val;
        }
        return result + self.bias;
    }

    pub fn feedforward(self: NetworkLayer, x: []const f32) f32 {
        var first_result = self.multiply_add(x);
        return sigmoid(first_result);
    }
};

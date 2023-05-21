const std = @import("std");

fn sigmoid(val: f32) f32 {
    return 1 / (1 + std.math.exp(-val));
}

pub const NetworkLayer = struct {
    weights: []f32,
    bias: f32,

    pub fn size(self: NetworkLayer) usize {
        return self.weights.len;
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

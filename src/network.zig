const std = @import("std");
const mat = @import("matrix.zig");
const maths = @import("maths.zig");

pub const NetworkLayer = struct {
    weights: mat.Matrix,
    biases: []f32,

    /// Returns size of the weight matrix
    /// (number of elements)
    pub fn size(self: NetworkLayer) usize {
        return self.weights.size();
    }

    pub fn feedforward(self: NetworkLayer, x: []const f32, out: []f32) void {
        self.weights.apply(&x, &out);
        mat.accumulate(&out, &self.biases);
        maths.apply_sigmoid_in_place(&out);
    }
};

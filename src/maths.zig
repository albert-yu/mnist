const std = @import("std");

fn sigmoid(val: f32) f32 {
    return 1 / (1 + std.math.exp(-val));
}

/// Applies sigmoid to each element, writing
/// to `out`
pub fn apply_sigmoid(arr: []f32, out: []f32) void {
    for (arr) |value, i| {
        out[i] = sigmoid(value);
    }
}

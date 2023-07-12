const std = @import("std");

fn sigmoid(val: f32) f32 {
    return 1 / (1 + std.math.exp(-val));
}

fn sigmoid_prime(val: f32) f32 {
    return sigmoid(val) / (1 - sigmoid(val));
}

fn apply(comptime operation: fn (f32) f32, arr: []f32, out: []f32) void {
    for (arr) |value, i| {
        out[i] = operation(value);
    }
}

/// Applies sigmoid to each element, writing
/// to `out`
pub fn apply_sigmoid(arr: []f32, out: []f32) void {
    apply(sigmoid, arr, out);
}

pub fn apply_sigmoid_in_place(arr: []f32) void {
    apply_sigmoid(arr, arr);
}

pub fn apply_sigmoid_prime(arr: []f32, out: []f32) void {
    apply(sigmoid_prime, arr, out);
}

pub fn apply_sigmoid_prime_in_place(arr: []f32) void {
    apply_sigmoid_prime(arr, arr);
}

const err_tolerance = 1e-9;

test "sigmoid test" {
    var vector = [_]f32{ 0, 1 };
    var out = [_]f32{0} ** 2;
    apply_sigmoid(&vector, &out);
    try std.testing.expectApproxEqRel(out[0], 0.5, err_tolerance);
    try std.testing.expectApproxEqRel(out[1], 0.731058578630074, err_tolerance);
}

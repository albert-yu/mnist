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

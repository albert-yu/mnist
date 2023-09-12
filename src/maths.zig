const std = @import("std");

pub fn sigmoid(val: f64) f64 {
    return 1 / (1 + std.math.exp(-val));
}

pub fn sigmoid_prime(val: f64) f64 {
    return sigmoid(val) * (1 - sigmoid(val));
}

const err_tolerance = 1e-9;

test "sigmoid test" {
    try std.testing.expectApproxEqRel(sigmoid(0), 0.5, err_tolerance);
    try std.testing.expectApproxEqRel(sigmoid(1), 0.731058578630074, err_tolerance);
}

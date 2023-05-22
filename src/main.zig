const std = @import("std");
const mat = @import("matrix.zig");
const maths = @import("maths.zig");
// const nn = @import("network.zig");

pub fn main() !void {
    // do nothing
}

test "matrix application test" {
    const err_tolerance = 1e-9;
    var matrix_data = [_]f32{
        1, 2, 1,
        4, 3, 4,
    };
    var matrix = mat.Matrix{
        .data = &matrix_data,
        .rows = 2,
        .cols = 3,
    };
    var vec = [_]f32{ 3, 2, 1 };
    var result = [_]f32{0} ** 2;
    matrix.apply(&vec, &result);
    // (8 22)^T
    var expected_0: f32 = 8;
    var expected_1: f32 = 22;
    try std.testing.expectApproxEqRel(expected_0, result[0], err_tolerance);
    try std.testing.expectApproxEqRel(expected_1, result[1], err_tolerance);
}

test "accumulate test" {
    var vector = [_]f32{ 1, 2 };
    var addend = [_]f32{ 2, 3 };
    mat.accumulate(&vector, &addend);
    const err_tolerance = 1e-9;
    var expected_0: f32 = 3;
    var expected_1: f32 = 5;
    try std.testing.expectApproxEqRel(expected_0, vector[0], err_tolerance);
    try std.testing.expectApproxEqRel(expected_1, vector[1], err_tolerance);
}

test "sigmoid test" {
    const err_tolerance = 1e-9;
    var vector = [_]f32{ 0, 1 };
    var out = [_]f32{0} ** 2;
    maths.apply_sigmoid(&vector, &out);
    try std.testing.expectApproxEqRel(out[0], 0.5, err_tolerance);
    try std.testing.expectApproxEqRel(out[1], 0.731058578630074, err_tolerance);
}

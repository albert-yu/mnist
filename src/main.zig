const std = @import("std");
const nn = @import("network.zig");

pub fn main() !void {
    // do nothing
}

test "multiply_add test" {
    const err_tolerance = 1e-9;
    const LAYER_SIZE = 3;
    var test_x = [LAYER_SIZE]f32{ 1, 2, 3 };
    var weights = [LAYER_SIZE]f32{ 1, 2, 3 };
    var test_layer = nn.NetworkLayer{ .weights = &weights, .bias = 0.1443 };
    var result = test_layer.multiply_add(&test_x);
    var expected: f32 = 14.1443;
    try std.testing.expectApproxEqRel(expected, result, err_tolerance);
}

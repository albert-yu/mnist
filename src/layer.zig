const std = @import("std");
const linalg = @import("linalg.zig");
const maths = @import("maths.zig");

/// Single layer in neural net
pub const Layer = struct {
    weights: linalg.Matrix,
    biases: linalg.Matrix,

    const Self = @This();

    pub fn alloc(allocator: std.mem.Allocator, comptime IN: usize, comptime OUT: usize) !Self {
        var w_buffer = try allocator.alloc(f64, IN * OUT);
        var b_buffer = try allocator.alloc(f64, OUT);
        return Self{ .weights = linalg.Matrix{
            .data = w_buffer,
            .rows = IN,
            .cols = OUT,
        }, .biases = linalg.Matrix{
            .data = b_buffer,
            .rows = IN,
            .cols = 1,
        } };
    }

    pub fn forward(self: Self, allocator: std.mem.Allocator, input: linalg.Matrix) !linalg.Matrix {
        var result = try self.weights.mult_alloc(allocator, input);
        try result.add(self.biases, &result);
        result.for_each(maths.sigmoid);
        return result;
    }

    pub fn init_randn(self: Self) void {
        const randgen = std.rand.DefaultPrng;
        var rand = randgen.init(1);
        for (self.weights.data) |_, i| {
            self.weights.data[i] = rand.random().floatNorm(f64);
        }
        for (self.biases.data) |_, i| {
            self.biases.data[i] = rand.random().floatNorm(f64);
        }
    }

    pub fn dealloc(self: Self, allocator: std.mem.Allocator) void {
        allocator.free(self.biases.data);
        allocator.free(self.weights.data);
    }
};

test "feedforward test" {
    const allocator = std.testing.allocator;
    var layer1 = try Layer.alloc(allocator, 2, 2);
    defer layer1.dealloc(allocator);
    var layer2 = try Layer.alloc(allocator, 2, 2);
    defer layer2.dealloc(allocator);

    var w_1 = [_]f64{
        1, 0,
        0, 1,
    };
    var b_1 = [_]f64{
        0.5,
        0.5,
    };
    layer1.weights.copy_data_unsafe(&w_1);
    layer1.biases.copy_data_unsafe(&b_1);

    var w_2 = [_]f64{
        -1, 0,
        0,  1,
    };
    var b_2 = [_]f64{
        0.2,
        0.2,
    };

    layer2.weights.copy_data_unsafe(&w_2);
    layer2.biases.copy_data_unsafe(&b_2);
    var input_x = [_]f64{
        0.1,
        0.1,
    };
    var input = linalg.Matrix{
        .data = &input_x,
        .rows = 2,
        .cols = 1,
    };
    const TOLERANCE = 1e-9;

    var result1 = try layer1.forward(allocator, input);
    defer result1.dealloc_data(allocator);

    var result2 = try layer2.forward(allocator, result1);
    defer result2.dealloc_data(allocator);

    // var output = network.output_layer();
    var expected_out = [_]f64{ 0.3903940131009935, 0.6996551604890665 };
    var activation = result2;
    try std.testing.expectApproxEqRel(expected_out[0], activation.data[0], TOLERANCE);
    try std.testing.expectApproxEqRel(expected_out[1], activation.data[1], TOLERANCE);
}

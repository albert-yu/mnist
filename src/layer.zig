const std = @import("std");
const linalg = @import("linalg.zig");
const maths = @import("maths.zig");

pub fn Gradients(comptime IN: usize, comptime OUT: usize) type {
    return struct {
        weights: linalg.Matrix,
        biases: linalg.Matrix,

        const Self = @This();

        pub fn alloc(allocator: std.mem.Allocator) !Self {
            var w_buffer = try allocator.alloc(f64, IN * OUT);
            var b_buffer = try allocator.alloc(f64, OUT);
            return Self{ .weights = linalg.Matrix{
                .data = w_buffer,
                .rows = IN,
                .cols = OUT,
            }, .biases = linalg.Matrix{
                .data = b_buffer,
                .rows = OUT,
                .cols = 1,
            } };
        }

        pub fn dealloc(self: Self, allocator: std.mem.Allocator) void {
            allocator.free(self.biases.data);
            allocator.free(self.weights.data);
        }
    };
}

/// Single layer in neural net
pub fn Layer(comptime IN: usize, comptime OUT: usize) type {
    return struct {
        weights: linalg.Matrix,
        biases: linalg.Matrix,
        last_input: linalg.Matrix,
        last_z: linalg.Matrix,

        const Self = @This();

        pub fn alloc(allocator: std.mem.Allocator) !Self {
            var w_buffer = try allocator.alloc(f64, IN * OUT);
            var b_buffer = try allocator.alloc(f64, OUT);
            var last_z_data = try allocator.alloc(f64, OUT);
            return Self{ .last_input = undefined, .weights = linalg.Matrix{
                .data = w_buffer,
                .rows = IN,
                .cols = OUT,
            }, .biases = linalg.Matrix{
                .data = b_buffer,
                .rows = OUT,
                .cols = 1,
            }, .last_z = linalg.Matrix{
                .data = last_z_data,
                .rows = OUT,
                .cols = 1,
            } };
        }

        pub fn forward(self: *Self, allocator: std.mem.Allocator, input: linalg.Matrix, comptime activation_fn: fn (f64) f64) !linalg.Matrix {
            var result = try self.weights.mult_alloc(allocator, input);
            try result.add(self.biases, &result);
            result.for_each(activation_fn);
            self.last_input = input;
            self.last_z.copy_data_unsafe(result.data);
            return result;
        }

        pub fn backward(self: Self, allocator: std.mem.Allocator, err: linalg.Matrix, comptime activation_prime: fn (f64) f64) !Gradients(IN, OUT) {
            var gradient_results = try Gradients(IN, OUT).alloc(allocator);
            var z_changes = try self.last_z.make_copy(allocator);
            defer z_changes.destroy(allocator);
            z_changes.for_each(activation_prime);
            z_changes.hadamard(err, &gradient_results.biases);

            var last_input_t = try self.last_input.t_alloc(allocator);
            defer last_input_t.dealloc_data(allocator);
            try gradient_results.biases.multiply(last_input_t, &gradient_results.weights);
            return gradient_results;
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

        pub fn init_zeros(self: Self) void {
            for (self.weights.data) |_, i| {
                self.weights.data[i] = 0;
            }
            for (self.biases.data) |_, i| {
                self.biases.data[i] = 0;
            }
        }

        pub fn dealloc(self: Self, allocator: std.mem.Allocator) void {
            self.weights.dealloc_data(allocator);
            self.biases.dealloc_data(allocator);
            self.last_z.dealloc_data(allocator);
        }
    };
}

test "feedforward test" {
    const allocator = std.testing.allocator;
    var layer1 = try Layer(2, 2).alloc(allocator);
    defer layer1.dealloc(allocator);
    var layer2 = try Layer(2, 2).alloc(allocator);
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

    var result1 = try layer1.forward(allocator, input, maths.sigmoid);
    defer result1.dealloc_data(allocator);

    var result2 = try layer2.forward(allocator, result1, maths.sigmoid);
    defer result2.dealloc_data(allocator);

    // var output = network.output_layer();
    var expected_out = [_]f64{ 0.3903940131009935, 0.6996551604890665 };
    var activation = result2;
    try std.testing.expectApproxEqRel(expected_out[0], activation.data[0], TOLERANCE);
    try std.testing.expectApproxEqRel(expected_out[1], activation.data[1], TOLERANCE);
}

test "backpropagation test" {
    const allocator = std.testing.allocator;
    var x_data = [_]f64{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 18, 18, 18, 126, 136, 175, 26, 166, 255, 247, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 36, 94, 154, 170, 253, 253, 253, 253, 253, 225, 172, 253, 242, 195, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 49, 238, 253, 253, 253, 253, 253, 253, 253, 253, 251, 93, 82, 82, 56, 39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 219, 253, 253, 253, 253, 253, 198, 182, 247, 241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 156, 107, 253, 253, 205, 11, 0, 43, 154, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 1, 154, 253, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 139, 253, 190, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 190, 253, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 241, 225, 160, 108, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 240, 253, 253, 119, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 186, 253, 253, 150, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 93, 252, 253, 187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 249, 253, 249, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 130, 183, 253, 253, 207, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 148, 229, 253, 253, 253, 250, 182, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 114, 221, 253, 253, 253, 253, 201, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 66, 213, 253, 253, 253, 253, 198, 81, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 171, 219, 253, 253, 253, 253, 195, 80, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 172, 226, 253, 253, 253, 253, 244, 133, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 136, 253, 253, 253, 212, 135, 132, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    var x = linalg.Matrix{
        .data = &x_data,
        .cols = 1,
        .rows = x_data.len,
    };

    var y_data = [_]f64{ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 };
    var y = linalg.Matrix{ .data = &y_data, .cols = 1, .rows = y_data.len };
    const image_size = 28 * 28;
    const HIDDEN_LAYER_SIZE = 30;
    const DIGITS = 10;
    var hidden_layer = try Layer(image_size, HIDDEN_LAYER_SIZE).alloc(allocator);
    defer hidden_layer.dealloc(allocator);
    hidden_layer.init_zeros();

    var output_layer = try Layer(HIDDEN_LAYER_SIZE, DIGITS).alloc(allocator);
    defer output_layer.dealloc(allocator);
    output_layer.init_zeros();

    // feedforward
    const result1 = try hidden_layer.forward(allocator, x, maths.sigmoid);
    defer result1.dealloc_data(allocator);

    const result2 = try output_layer.forward(allocator, result1, maths.sigmoid);
    defer result2.dealloc_data(allocator);

    // compute error
    var err = try result2.sub_alloc(allocator, y);
    defer err.dealloc_data(allocator);

    // backward
    var grad1 = try output_layer.backward(allocator, err, maths.sigmoid_prime);
    defer grad1.dealloc(allocator);

    // const layer_sizes = [_]usize{ image_size, HIDDEN_LAYER_SIZE, DIGITS };
    // var network = try alloc_network(allocator, &layer_sizes);
    // defer free_network(allocator, network);
    // network.init_zeros();
    // const result = try network.backprop(allocator, data_point);

    var expected_delta_b = [_]f64{ 0.125, 0.125, 0.125, 0.125, 0.125, -0.125, 0.125, 0.125, 0.125, 0.125 };
    // try std.testing.expectEqualSlices(f64, &expected_delta_b, result.delta_nabla_biases[1].data);
    try std.testing.expectEqualSlices(f64, &expected_delta_b, grad1.biases.data);

    // expected weights for reference
    //     [[ 0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625]
    //  [ 0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625]
    //  [ 0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625]
    //  [ 0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625]
    //  [ 0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625]
    //  [-0.0625 -0.0625 -0.0625 -0.0625 -0.0625 -0.0625 -0.0625 -0.0625 -0.0625
    //   -0.0625 -0.0625 -0.0625 -0.0625 -0.0625 -0.0625 -0.0625 -0.0625 -0.0625
    //   -0.0625 -0.0625 -0.0625 -0.0625 -0.0625 -0.0625 -0.0625 -0.0625 -0.0625
    //   -0.0625 -0.0625 -0.0625]
    //  [ 0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625]
    //  [ 0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625]
    //  [ 0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625]
    //  [ 0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625  0.0625
    //    0.0625  0.0625  0.0625]]
}

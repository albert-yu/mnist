const std = @import("std");
const lin = @import("linalg.zig");
const maths = @import("maths.zig");
// const nn = @import("network.zig");

/// https://stackoverflow.com/a/73020142
fn range(len: usize) []const void {
    return @as([*]void, undefined)[0..len];
}

fn get_double_word(bytes: []u8, offset: usize) u32 {
    const slice = bytes[offset .. offset + 4][0..4];
    return std.mem.readInt(u32, slice, std.builtin.Endian.Big);
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const file = try std.fs.cwd().openFile("data/train-labels.idx1-ubyte", .{});
    defer file.close();

    const size = try file.getEndPos();
    var buffer = try allocator.alloc(u8, size);
    defer allocator.free(buffer);
    _ = try file.read(buffer);

    // read first labels
    const count_offset = 4;
    const label_count = get_double_word(buffer, count_offset);

    std.debug.print("label count: {}\n", .{label_count});
    const start_index = 8;

    for (range(label_count)) |_, i| {
        if (i >= 20) {
            break;
        }
        const val = buffer[start_index + i];
        std.debug.print("{}\n", .{val});
    }
}

const err_tolerance = 1e-9;

test "matrix application test" {
    var matrix_data = [_]f32{
        1, 2, 1,
        4, 3, 4,
    };
    var matrix = lin.Matrix{
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
    lin.accumulate(&vector, &addend);
    var expected_0: f32 = 3;
    var expected_1: f32 = 5;
    try std.testing.expectApproxEqRel(expected_0, vector[0], err_tolerance);
    try std.testing.expectApproxEqRel(expected_1, vector[1], err_tolerance);
}

test "transpose test" {
    var matrix_data = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    var matrix: lin.Matrix = .{
        .data = &matrix_data,
        .rows = 2,
        .cols = 3,
    };
    var t_matrix_init = [_]f32{0} ** matrix_data.len;
    var t_matrix = lin.Matrix{
        .data = &t_matrix_init,
        .rows = 0,
        .cols = 0,
    };
    lin.transpose(matrix, &t_matrix);
    var expected_rows: usize = 3;
    var expected_cols: usize = 2;
    try std.testing.expectEqual(expected_rows, t_matrix.rows);
    try std.testing.expectEqual(expected_cols, t_matrix.cols);
    var result_data = [_]f32{
        1, 4,
        2, 5,
        3, 6,
    };
    try std.testing.expectEqualSlices(f32, &result_data, t_matrix.data);
}

test "sigmoid test" {
    var vector = [_]f32{ 0, 1 };
    var out = [_]f32{0} ** 2;
    maths.apply_sigmoid(&vector, &out);
    try std.testing.expectApproxEqRel(out[0], 0.5, err_tolerance);
    try std.testing.expectApproxEqRel(out[1], 0.731058578630074, err_tolerance);
}

test "matrix multiplication test" {
    const mat_t = f32;
    var data = [_]mat_t{
        1, 2, 3,
        3, 1, 4,
    };
    var data_other = [_]mat_t{
        1, 1,
        2, 1,
        2, 5,
    };
    var matrix = lin.Matrix{
        .data = &data,
        .rows = 2,
        .cols = 3,
    };

    var matrix_other = lin.Matrix{
        .data = &data_other,
        .rows = 3,
        .cols = 2,
    };
    var out_data = [_]mat_t{0} ** 4;
    var out_matrix = lin.Matrix{
        .data = &out_data,
        .rows = 2,
        .cols = 2,
    };

    try matrix.multiply(matrix_other, &out_matrix);
    var expected_out_data = [_]mat_t{
        11, 18,
        13, 24,
    };
    try std.testing.expectEqualSlices(mat_t, &expected_out_data, out_matrix.data);
}

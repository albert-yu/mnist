const std = @import("std");

/// Add `vec1` and `vec2`, store result in `out`
fn sum(vec1: []f64, vec2: []const f64, out: []f64) void {
    for (vec1, 0..) |val, i| {
        const other_val = vec2[i];
        out[i] = val + other_val;
    }
}

/// Subtract `vec2` from `vec1`, store result in `out`
fn subtract(vec1: []f64, vec2: []const f64, out: []f64) void {
    for (vec1, 0..) |val, i| {
        const other_val = vec2[i];
        out[i] = val - other_val;
    }
}

/// Computes Hadamard product (element-wise multiplication)
///
/// Assumes `out` is allocated to be the same length as both
/// `vec1` and `vec2`.
fn hadamard_product(vec1: []f64, vec2: []f64, out: []f64) void {
    for (vec1, 0..) |el1, i| {
        const el2 = vec2[i];
        out[i] = el1 * el2;
    }
}

/// Sets the resulting transposed matrix
/// to `out`.
///
/// In-place transposition is a non-trivial problem:
/// https://en.wikipedia.org/wiki/In-place_matrix_transposition
fn transpose(in: Matrix, out: *Matrix) void {
    // swap rows and columns
    out.rows = in.cols;
    out.cols = in.rows;
    var i: usize = 0;
    while (i < in.rows) : (i += 1) {
        var j: usize = 0;
        while (j < in.cols) : (j += 1) {
            out.set(j, i, in.at(i, j));
        }
    }
}

pub const Matrix = struct {
    data: []f64,
    rows: usize,
    cols: usize,

    const Self = @This();

    pub fn new(rows: usize, cols: usize) Self {
        return Self{
            .rows = rows,
            .cols = cols,
            .data = undefined,
        };
    }

    pub fn alloc(self: *Self, allocator: std.mem.Allocator) !void {
        self.data = try allocator.alloc(f64, self.rows * self.cols);
    }

    pub fn dealloc(self: Self, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
    }

    pub fn mul_alloc(self: Self, allocator: std.mem.Allocator, right: Self) !Self {
        var result = Matrix.new(self.cols, right.rows);
        try result.alloc(allocator);
        self.multiply(right, &result);
        return result;
    }

    /// Number of elements in this matrix
    pub inline fn size(self: Self) usize {
        return self.data.len;
    }

    pub fn print(self: Matrix) void {
        for (self.data, 0..) |el, i| {
            if (i % self.cols == 0) {
                std.debug.print("\n", .{});
            }
            std.debug.print("{} ", .{el});
        }
        std.debug.print("\n", .{});
    }

    pub fn add(self: Matrix, other: Matrix, out: *Matrix) void {
        out.rows = self.rows;
        out.cols = self.cols;
        sum(self.data, other.data, out.data);
    }

    pub fn sub(self: Matrix, other: Matrix, out: *Matrix) error{MatrixDimensionError}!void {
        if (self.cols != other.cols or self.rows != other.rows) {
            return error.MatrixDimensionError;
        }
        out.rows = self.rows;
        out.cols = self.cols;
        subtract(self.data, other.data, out.data);
    }

    /// Sets all elements to 0
    pub fn zeroes(self: Matrix) void {
        for (self.data, 0..) |_, i| {
            self.data[i] = 0;
        }
    }

    /// scales all matrix elements in-place
    pub fn scale(self: Matrix, scalar: f64) void {
        for (self.data, 0..) |elem, i| {
            self.data[i] = elem * scalar;
        }
    }

    pub fn multiply_unsafe(self: Matrix, right: Matrix, out: *Matrix) void {
        out.rows = self.rows;
        out.cols = right.cols;
        var i: usize = 0;
        while (i < out.rows) : (i += 1) {
            var j: usize = 0;
            while (j < out.cols) : (j += 1) {
                var acc: f64 = 0;
                var k: usize = 0;
                while (k < self.cols) : (k += 1) {
                    acc += self.at(i, k) * right.at(k, j);
                }
                out.set(i, j, acc);
            }
        }
    }

    /// Multiples two matrices, stores result in `out`.
    /// Assumes `out` is properly allocated, but will set
    /// the correct rows and cols.
    pub fn multiply(self: Matrix, right: Matrix, out: *Matrix) error{MatrixDimensionError}!void {
        if (self.cols != right.rows) {
            return error.MatrixDimensionError;
        }
        self.multiply_unsafe(right, out);
    }

    pub fn dealloc_data(self: Matrix, allocator: std.mem.Allocator) void {
        free_matrix_data(allocator, self);
    }

    pub fn mult_alloc(self: Matrix, allocator: std.mem.Allocator, right: Matrix) !Matrix {
        var result = Matrix{ .rows = 0, .cols = 0, .data = undefined };
        try alloc_matrix_data(allocator, &result, self.rows, right.cols);
        try self.multiply(right, &result);
        return result;
    }

    pub fn sub_alloc(self: Matrix, allocator: std.mem.Allocator, right: Matrix) !Matrix {
        var result = Matrix{ .rows = 0, .cols = 0, .data = undefined };
        try alloc_matrix_data(allocator, &result, self.rows, right.cols);
        try self.sub(right, &result);
        return result;
    }

    /// Transposes matrix and returns new one, which must be dealloc'd
    pub fn t_alloc(self: Matrix, allocator: std.mem.Allocator) !Matrix {
        var result = Matrix{ .rows = 0, .cols = 0, .data = undefined };
        try alloc_matrix_data(allocator, &result, self.cols, self.rows);
        transpose(self, &result);
        return result;
    }

    pub fn hadamard(self: Matrix, other: Matrix, out: *Matrix) void {
        hadamard_product(self.data, other.data, out.data);
    }

    pub fn for_each(self: Matrix, comptime op: fn (f64) f64) void {
        for (self.data, 0..) |_, i| {
            self.data[i] = op(self.data[i]);
        }
    }

    /// Maps 2D indices to 1D underlying offset
    inline fn get_offset(self: Matrix, i: usize, j: usize) usize {
        return i * self.cols + j;
    }

    /// Returns the value at the given indices.
    ///
    /// Parameters:
    ///   i - 0-based row index
    ///   j - 0-based column index
    pub inline fn at(self: Matrix, i: usize, j: usize) f64 {
        var index = self.get_offset(i, j);
        return self.data[index];
    }

    /// Sets the value at the given indices.
    ///
    /// Parameters:
    ///   i - 0-based row index
    ///   j - 0-based column index
    ///   value - value to set
    pub fn set(self: Matrix, i: usize, j: usize, value: f64) void {
        var index = self.get_offset(i, j);
        self.data[index] = value;
    }

    /// Copies the input data into its own data buffer
    /// without checking bounds
    pub fn copy_data_unsafe(self: Matrix, data: []f64) void {
        for (data, 0..) |elem, i| {
            self.data[i] = elem;
        }
    }

    pub fn make_copy(self: Matrix, allocator: std.mem.Allocator) !*Matrix {
        var copied = try alloc_matrix_with_values(allocator, self.rows, self.cols, self.data);
        return copied;
    }
};

pub fn alloc_matrix_data(allocator: std.mem.Allocator, matrix: *Matrix, rows: usize, cols: usize) error{OutOfMemory}!void {
    matrix.data = try allocator.alloc(f64, rows * cols);
    matrix.rows = rows;
    matrix.cols = cols;
}

pub fn free_matrix_data(allocator: std.mem.Allocator, matrix: Matrix) void {
    allocator.free(matrix.data);
}

pub fn alloc_matrix(allocator: std.mem.Allocator, rows: usize, cols: usize) error{OutOfMemory}!*Matrix {
    var matrix = try allocator.create(Matrix);
    try alloc_matrix_data(allocator, matrix, rows, cols);
    return matrix;
}

/// Copies the data input into allocated memory
pub fn alloc_matrix_with_values(allocator: std.mem.Allocator, rows: usize, cols: usize, data: []const f64) error{ DimensionsMismatch, OutOfMemory }!*Matrix {
    if (rows * cols != data.len) {
        return error.DimensionsMismatch;
    }
    var matrix = try alloc_matrix(allocator, rows, cols);
    for (data, 0..) |val, i| {
        matrix.data[i] = val;
    }
    return matrix;
}

pub fn free_matrix(allocator: std.mem.Allocator, matrix: *Matrix) void {
    free_matrix_data(allocator, matrix.*);
    allocator.destroy(matrix);
}

pub fn matrix_multiply(allocator: std.mem.Allocator, matrix_left: Matrix, matrix_right: Matrix) error{ MatrixDimensionError, OutOfMemory }!*Matrix {
    var out_matrix = try alloc_matrix(allocator, matrix_left.rows, matrix_right.cols);
    try matrix_left.multiply(matrix_right, out_matrix);
    return out_matrix;
}

pub fn matrix_copy(allocator: std.mem.Allocator, matrix: Matrix) !*Matrix {
    var result = try alloc_matrix_with_values(allocator, matrix.rows, matrix.cols, matrix.data);
    return result;
}

const err_tolerance = 1e-9;

test "transpose test" {
    var matrix_data = [_]f64{
        1, 2, 3,
        4, 5, 6,
    };
    var matrix: Matrix = .{
        .data = &matrix_data,
        .rows = 2,
        .cols = 3,
    };
    var t_matrix_init = [_]f64{0} ** matrix_data.len;
    var t_matrix = Matrix{
        .data = &t_matrix_init,
        .rows = 0,
        .cols = 0,
    };
    transpose(matrix, &t_matrix);
    var expected_rows: usize = 3;
    var expected_cols: usize = 2;
    try std.testing.expectEqual(expected_rows, t_matrix.rows);
    try std.testing.expectEqual(expected_cols, t_matrix.cols);
    var result_data = [_]f64{
        1, 4,
        2, 5,
        3, 6,
    };
    try std.testing.expectEqualSlices(f64, &result_data, t_matrix.data);
}

test "matrix multiplication test" {
    const mat_t = f64;
    var data = [_]mat_t{
        1, 2, 3,
        3, 1, 4,
    };
    var data_other = [_]mat_t{
        1, 1,
        2, 1,
        2, 5,
    };
    var matrix = Matrix{
        .data = &data,
        .rows = 2,
        .cols = 3,
    };

    var matrix_other = Matrix{
        .data = &data_other,
        .rows = 3,
        .cols = 2,
    };
    var out_data = [_]mat_t{0} ** 4;
    var out_matrix = Matrix{
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

test "outer product test" {
    const mat_t = f64;
    var data = [_]mat_t{
        1,
        2,
        3,
        4,
    };
    var data_other = [_]mat_t{
        1, 2, 3,
    };
    var matrix = Matrix{
        .data = &data,
        .rows = data.len,
        .cols = 1,
    };

    var matrix_other = Matrix{
        .data = &data_other,
        .rows = 1,
        .cols = data_other.len,
    };
    comptime var total_size = 4 * 3;
    var out_data = [_]mat_t{0} ** total_size;
    var out_matrix = Matrix{
        .data = &out_data,
        .rows = 4,
        .cols = 3,
    };

    try matrix.multiply(matrix_other, &out_matrix);
    var expected_out_data = [_]mat_t{
        1, 2, 3,
        2, 4, 6,
        3, 6, 9,
        4, 8, 12,
    };
    try std.testing.expectEqualSlices(mat_t, &expected_out_data, out_matrix.data);
}

test "inner product test" {
    var data_a_t = [_]f64{
        1, 2, 3,
    };
    var a_t = Matrix{
        .data = &data_a_t,
        .rows = 1,
        .cols = data_a_t.len,
    };
    var a = Matrix{
        .data = &data_a_t,
        .rows = data_a_t.len,
        .cols = 1,
    };
    var out_data = [_]f64{0};
    var out = Matrix{
        .data = &out_data,
        .rows = 1,
        .cols = 1,
    };
    try a_t.multiply(a, &out);
    var expected_out_data = [_]f64{14};
    try std.testing.expectEqualSlices(f64, &expected_out_data, out.data);
}

test "linear transform test with allocation" {
    const allocator = std.testing.allocator;
    var identity_matrix_data = [_]f64{
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    };
    const identity_matrix = try alloc_matrix_with_values(allocator, 3, 3, &identity_matrix_data);
    defer free_matrix(allocator, identity_matrix);

    var vector_data = [_]f64{
        4,
        0,
        223,
    };
    const some_vector = try alloc_matrix_with_values(allocator, 3, 1, &vector_data);
    defer free_matrix(allocator, some_vector);

    const result = try matrix_multiply(allocator, identity_matrix.*, some_vector.*);
    defer free_matrix(allocator, result);

    try std.testing.expectEqualSlices(f64, result.data, some_vector.data);
}

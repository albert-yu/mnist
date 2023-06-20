const std = @import("std");
/// Computes inner (dot) product.
///
/// Input vectors assumed to be of equal length.
pub fn inner_product(vec1: []f32, vec2: []f32) f32 {
    var result: f32 = 0;
    for (vec1) |val, i| {
        const other_val = vec2[i];
        result += val * other_val;
    }
    return result;
}

/// Add `vec1` and `vec2`, store result in `out`
pub fn sum(vec1: []f32, vec2: []const f32, out: []f32) void {
    for (vec1) |val, i| {
        const other_val = vec2[i];
        out[i] = val + other_val;
    }
}

/// Subtract `vec2` from `vec1`, store result in `out`
pub fn subtract(vec1: []f32, vec2: []const f32, out: []f32) void {
    for (vec1) |val, i| {
        const other_val = vec2[i];
        out[i] = val - other_val;
    }
}

/// Computes Hadamard product (element-wise multiplication)
///
/// Assumes `out` is allocated to be the same length as both
/// `vec1` and `vec2`.
pub fn hadamard_product(vec1: []f32, vec2: []f32, out: []f32) void {
    for (vec1) |el1, i| {
        const el2 = vec2[i];
        out[i] = el1 * el2;
    }
}

/// Sets the resulting transposed matrix
/// to `out`.
///
/// In-place transposition is a non-trivial problem:
/// https://en.wikipedia.org/wiki/In-place_matrix_transposition
pub fn transpose(in: Matrix, out: *Matrix) void {
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

pub fn accumulate(acc: []f32, addend: []f32) void {
    sum(acc, addend, acc);
}

pub const Matrix = struct {
    data: []f32,
    rows: usize,
    cols: usize,

    /// Number of elements in this matrix
    pub fn size(self: Matrix) usize {
        return self.data.len;
    }

    pub fn num_rows(self: Matrix) usize {
        return self.rows;
    }

    pub fn num_cols(self: Matrix) usize {
        return self.cols;
    }

    pub fn print(self: Matrix) void {
        for (self.data) |el, i| {
            if (i % self.cols == 0) {
                std.debug.print("\n", .{});
            }
            std.debug.print("{} ", .{el});
        }
        std.debug.print("\n", .{});
    }

    /// Applies the matrix as a ar transformation
    /// to the vector (left multiplication),
    /// assuming correct dimensions.
    /// Writes the result to out
    pub fn apply(self: Matrix, vec: []const f32, out: []f32) void {
        var i: usize = 0;
        while (i < self.rows) : (i += 1) {
            var acc: f32 = 0;
            var j: usize = 0;
            while (j < self.cols) : (j += 1) {
                var value = self.at(i, j);
                var vec_value = vec[j];
                acc += value * vec_value;
            }
            out[i] = acc;
        }
    }

    /// Multiples two matrices, stores result in `out`.
    /// Assumes `out` is properly allocated, but will set
    /// the correct rows and cols.
    pub fn multiply(self: Matrix, other: Matrix, out: *Matrix) error{MatrixDimensionError}!void {
        if (self.num_cols() != other.num_rows()) {
            return error.MatrixDimensionError;
        }
        out.rows = self.num_rows();
        out.cols = other.num_cols();
        var i: usize = 0;
        while (i < out.num_rows()) : (i += 1) {
            var j: usize = 0;
            while (j < out.num_cols()) : (j += 1) {
                var acc: f32 = 0;
                var k: usize = 0;
                while (k < self.num_cols()) : (k += 1) {
                    acc += self.at(i, k) * other.at(k, j);
                }
                out.set(i, j, acc);
            }
        }
    }

    /// Maps 2D indices to 1D underlying offset
    fn get_offset(self: Matrix, i: usize, j: usize) usize {
        return i * self.cols + j;
    }

    /// Returns the value at the given indices.
    ///
    /// Parameters:
    ///   i - 0-based row index
    ///   j - 0-based column index
    pub fn at(self: Matrix, i: usize, j: usize) f32 {
        var index = self.get_offset(i, j);
        return self.data[index];
    }

    /// Sets the value at the given indices.
    ///
    /// Parameters:
    ///   i - 0-based row index
    ///   j - 0-based column index
    ///   value - value to set
    pub fn set(self: Matrix, i: usize, j: usize, value: f32) void {
        var index = self.get_offset(i, j);
        self.data[index] = value;
    }
};

const err_tolerance = 1e-9;

test "matrix application test" {
    var matrix_data = [_]f32{
        1, 2, 1,
        4, 3, 4,
    };
    var matrix = Matrix{
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
    accumulate(&vector, &addend);
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
    var matrix: Matrix = .{
        .data = &matrix_data,
        .rows = 2,
        .cols = 3,
    };
    var t_matrix_init = [_]f32{0} ** matrix_data.len;
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
    var result_data = [_]f32{
        1, 4,
        2, 5,
        3, 6,
    };
    try std.testing.expectEqualSlices(f32, &result_data, t_matrix.data);
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
    const mat_t = f32;
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
    var data_a_t = [_]f32{
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
    var out_data = [_]f32{0};
    var out = Matrix{
        .data = &out_data,
        .rows = 1,
        .cols = 1,
    };
    try a_t.multiply(a, &out);
    var expected_out_data = [_]f32{14};
    try std.testing.expectEqualSlices(f32, &expected_out_data, out.data);
}

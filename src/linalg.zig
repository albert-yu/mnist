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
pub fn sum(vec1: []f32, vec2: []f32, out: []f32) void {
    for (vec1) |val, i| {
        const other_val = vec2[i];
        out[i] = val + other_val;
    }
}

/// Subtract `vec2` from `vec1`, store result in `out`
pub fn subtract(vec1: []f32, vec2: []f32, out: []f32) void {
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

    /// Applies the matrix as a linear transformation
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

    pub fn copy_col(self: Matrix, index: usize, out: []f32) void {
        if (index > self.num_cols()) {
            // silent no-op
            return;
        }
        var row_index: usize = 0;
        while (row_index < self.num_rows()) : (row_index += 1) {
            out[row_index] = self.at(row_index, index);
        }
    }

    pub fn set_col(self: Matrix, index: usize, in: []f32) void {
        if (index > self.num_rows()) {
            return;
        }
        var row_index: usize = 0;
        while (row_index < self.num_rows()) : (row_index += 1) {
            self.set(row_index, index, in[row_index]);
        }
    }

    /// Multiples two matrices, stores result in `out`.
    /// Assumes `out` is properly allocated.
    pub fn multiply(self: Matrix, other: Matrix, out: Matrix) error{ MatrixDimensionError, OutOfMemory }!void {
        if (self.num_cols() != other.num_rows()) {
            return error.MatrixDimensionError;
        }
        var i: usize = 0;
        var allocator = std.heap.page_allocator;
        // reuse buffer
        const vec = try allocator.alloc(f32, other.num_rows());
        const out_vec = try allocator.alloc(f32, other.num_rows());
        defer allocator.free(vec);
        defer allocator.free(out_vec);
        while (i < other.num_cols()) : (i += 1) {
            other.copy_col(i, vec);
            self.apply(vec, out_vec);
            out.set_col(i, out_vec);
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

const std = @import("std");

const VECTOR_SIZE = 8;

const Vec8 = @Vector(VECTOR_SIZE, f64);

fn aligned_calloc(allocator: std.mem.Allocator, size: usize) ![]Vec8 {
    const ptr = try allocator.alloc(Vec8, size);
    @memset(ptr, Vec8{ 0, 0, 0, 0, 0, 0, 0, 0 });
    return ptr;
}

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
        var result = Self.new(self.rows, right.cols);

        try result.alloc(allocator);
        try self.multiply(allocator, right, &result);
        return result;
    }

    /// Number of elements in this matrix
    pub inline fn size(self: Self) usize {
        return self.data.len;
    }

    pub fn print(self: Self) void {
        for (self.data, 0..) |el, i| {
            if (i % self.cols == 0) {
                std.debug.print("\n", .{});
            }
            std.debug.print("{} ", .{el});
        }
        std.debug.print("\n", .{});
    }

    pub fn add(self: Self, other: Self, out: *Self) void {
        out.rows = self.rows;
        out.cols = self.cols;
        sum(self.data, other.data, out.data);
    }

    pub fn sub(self: Self, other: Self, out: *Self) void {
        out.rows = self.rows;
        out.cols = self.cols;
        subtract(self.data, other.data, out.data);
    }

    /// Sets all elements to 0
    pub fn zeroes(self: Self) void {
        for (self.data, 0..) |_, i| {
            self.data[i] = 0;
        }
    }

    /// scales all matrix elements in-place
    pub fn scale(self: Self, scalar: f64) void {
        for (self.data, 0..) |elem, i| {
            self.data[i] = elem * scalar;
        }
    }

    /// Multiply but like faster
    fn mul(self: Self, allocator: std.mem.Allocator, right: Self, out: *Self) !void {
        const left_rows_blocks = (self.rows + VECTOR_SIZE - 1) / VECTOR_SIZE;
        var left_vec: []Vec8 = try aligned_calloc(allocator, left_rows_blocks * self.cols);
        defer allocator.free(left_vec);

        // transpose right matrix for better cache locality
        const right_t_rows = right.cols;
        const right_t_cols = right.rows;
        const right_t_cols_blocks = (right_t_rows + VECTOR_SIZE - 1) / VECTOR_SIZE;
        const right_t_vec: []Vec8 = try aligned_calloc(allocator, right_t_cols * right_t_cols_blocks);
        defer allocator.free(right_t_vec);

        // populate self matrix, row-major
        for (self.data, 0..) |elem, i| {
            const block_i = i / VECTOR_SIZE;
            const block_j = i % VECTOR_SIZE;
            left_vec[block_i][block_j] = elem;
        }

        // populate right_t values as vectors
        var rt_offset: usize = 0;
        for (0..(right.cols)) |j| {
            for (0..(right.rows)) |i| {
                const elem = right.at(i, j);
                const block_i = rt_offset / VECTOR_SIZE;
                const block_j = rt_offset % VECTOR_SIZE;
                right_t_vec[block_i][block_j] = elem;
                rt_offset += 1;
            }
        }
        std.debug.print("right_vec: {}\n\n", .{right_t_vec[0]});

        out.rows = self.rows;
        out.cols = right.cols;

        // perform the multiplication
        for (0..(out.rows)) |i| {
            for (0..(out.cols)) |j| {
                var acc = Vec8{ 0, 0, 0, 0, 0, 0, 0, 0 };
                for (0..(self.cols)) |k| {
                    const left_val = left_vec[i * left_rows_blocks + k];
                    const right_val = right_t_vec[k * right_t_cols_blocks + j];
                    acc += left_val * right_val;
                }

                for (0..VECTOR_SIZE) |k| {
                    out.set(i, j, out.at(i, j) + acc[k]);
                }
            }
        }
    }

    pub fn multiply(self: Self, allocator: std.mem.Allocator, right: Self, out: *Self) !void {
        // cannot inline this for the case
        // where out == &self (TODO: why?)
        try self.mul(allocator, right, out);
    }

    pub fn sub_alloc(self: Self, allocator: std.mem.Allocator, right: Self) !Self {
        var result = Self.new(self.rows, self.cols);
        try result.alloc(allocator);
        self.sub(right, &result);
        return result;
    }

    /// Transposes matrix and returns new one, which must be dealloc'd
    pub fn t_alloc(self: Self, allocator: std.mem.Allocator) !Self {
        var out = Self.new(self.cols, self.rows);
        try out.alloc(allocator);

        // swap rows and columns
        var i: usize = 0;
        while (i < self.rows) : (i += 1) {
            var j: usize = 0;
            while (j < self.cols) : (j += 1) {
                out.set(j, i, self.at(i, j));
            }
        }
        return out;
    }

    pub fn hadamard(self: Self, other: Self, out: *Self) void {
        hadamard_product(self.data, other.data, out.data);
    }

    pub fn for_each(self: Self, comptime op: fn (f64) f64) void {
        for (self.data, 0..) |_, i| {
            self.data[i] = op(self.data[i]);
        }
    }

    /// Maps 2D indices to 1D underlying offset
    inline fn get_offset(self: Self, i: usize, j: usize) usize {
        return i * self.cols + j;
    }

    /// Returns the value at the given indices.
    ///
    /// Parameters:
    ///   i - 0-based row index
    ///   j - 0-based column index
    pub inline fn at(self: Self, i: usize, j: usize) f64 {
        const index = self.get_offset(i, j);
        return self.data[index];
    }

    /// Sets the value at the given indices.
    ///
    /// Parameters:
    ///   i - 0-based row index
    ///   j - 0-based column index
    ///   value - value to set
    pub fn set(self: Self, i: usize, j: usize, value: f64) void {
        const index = self.get_offset(i, j);
        self.data[index] = value;
    }

    /// Copies the input data into its own data buffer
    /// without checking bounds
    pub fn set_data(self: Self, data: []f64) void {
        for (data, 0..) |elem, i| {
            self.data[i] = elem;
        }
    }

    pub fn make_copy(self: Self, allocator: std.mem.Allocator) !Self {
        var copied = Self.new(self.rows, self.cols);
        try copied.alloc(allocator);
        copied.set_data(self.data);
        return copied;
    }
};

const err_tolerance = 1e-9;

test "transpose test" {
    const allocator = std.testing.allocator;
    var matrix_data = [_]f64{
        1, 2, 3,
        4, 5, 6,
    };
    var matrix: Matrix = .{
        .data = &matrix_data,
        .rows = 2,
        .cols = 3,
    };
    const t_matrix = try matrix.t_alloc(allocator);
    defer t_matrix.dealloc(allocator);
    const expected_rows: usize = 3;
    const expected_cols: usize = 2;
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
    const allocator = std.testing.allocator;
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

    const matrix_other = Matrix{
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

    try matrix.multiply(allocator, matrix_other, &out_matrix);
    var expected_out_data = [_]mat_t{
        11, 18,
        13, 24,
    };
    try std.testing.expectEqualSlices(mat_t, &expected_out_data, out_matrix.data);
}

test "outer product test" {
    const mat_t = f64;
    const allocator = std.testing.allocator;
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

    const matrix_other = Matrix{
        .data = &data_other,
        .rows = 1,
        .cols = data_other.len,
    };

    var out_matrix = try matrix.mul_alloc(allocator, matrix_other);
    defer out_matrix.dealloc(allocator);
    var expected_out_data = [_]mat_t{
        1, 2, 3,
        2, 4, 6,
        3, 6, 9,
        4, 8, 12,
    };
    try std.testing.expectEqualSlices(mat_t, &expected_out_data, out_matrix.data);
}

test "inner product test" {
    const allocator = std.testing.allocator;
    var data_a_t = [_]f64{
        1, 2, 3,
    };
    var a_t = Matrix{
        .data = &data_a_t,
        .rows = 1,
        .cols = data_a_t.len,
    };
    const a = Matrix{
        .data = &data_a_t,
        .rows = data_a_t.len,
        .cols = 1,
    };
    var out = try a_t.mul_alloc(allocator, a);
    defer out.dealloc(allocator);
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
    var identity_matrix = Matrix.new(3, 3);
    try identity_matrix.alloc(allocator);
    defer identity_matrix.dealloc(allocator);
    identity_matrix.set_data(&identity_matrix_data);

    var vector_data = [_]f64{
        4,
        0,
        223,
    };
    var some_vector = Matrix.new(3, 1);
    try some_vector.alloc(allocator);
    defer some_vector.dealloc(allocator);
    some_vector.set_data(&vector_data);

    const result = try identity_matrix.mul_alloc(allocator, some_vector);
    defer result.dealloc(allocator);

    try std.testing.expectEqualSlices(f64, result.data, some_vector.data);
}

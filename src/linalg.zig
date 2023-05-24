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

pub fn sum(vec1: []f32, vec2: []f32, out: []f32) void {
    for (vec1) |val, i| {
        const other_val = vec2[i];
        out[i] = val + other_val;
    }
}

/// Sets the resulting transposed matrix
/// to `out`.
///
/// In-place transposition is a non-trivial problem:
/// https://en.wikipedia.org/wiki/In-place_matrix_transposition
pub fn transpose(in: *Matrix, out: *Matrix) void {
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

fn swap(arr: []f32, i: usize, j: usize) void {
    var val = arr[i];
    arr[i] = arr[j];
    arr[j] = val;
}

pub const Matrix = struct {
    data: []f32,
    rows: usize,
    cols: usize,

    /// Number of elements in this matrix
    pub fn size(self: Matrix) usize {
        return self.data.len;
    }

    /// Applies the matrix as a linear transformation
    /// to the vector (left multiplication),
    /// assuming correct dimensions.
    /// Writes the result to out
    pub fn apply(self: Matrix, vec: []f32, out: []f32) void {
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
